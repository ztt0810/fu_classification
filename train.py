import pandas as pd
import numpy as np

train_label=pd.read_csv('./data/train_label.csv')
len(train_label[train_label['label']==1.0]),len(train_label[train_label['label']==0.0]) #可见正负样本十分均衡


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import logging

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorboardX import SummaryWriter
# from cnn_finetune import make_model
from dataset import fuDataset

import warnings
warnings.filterwarnings("ignore")

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train_model(model,criterion, optimizer, lr_scheduler=None):

    train_dataset = fuDataset(opt.train_val_data, opt.train_label_csv, phase='train', input_size=opt.input_size)
    trainloader = DataLoader(train_dataset,
                             batch_size=opt.train_batch_size,
                             shuffle=True,
                             num_workers=opt.num_workers)

    total_iters=len(trainloader)
    logger.info('total_iters:{}'.format(total_iters))
    model_name=opt.backbone
    since = time.time()
    best_score = 0.0
    best_epoch = 0
    log_acc=0
    log_train=0
    writer = SummaryWriter()  # 用于记录训练和测试的信息:loss,acc等
    logger.info('start training...')
    #
    iters = len(trainloader)
    for epoch in range(1,opt.max_epoch+1):
        model.train(True)
        begin_time=time.time()
        logger.info('learning rate:{}'.format(optimizer.param_groups[-1]['lr']))
        logger.info('Epoch {}/{}'.format(epoch, opt.max_epoch))
        logger.info('-' * 10)
        running_corrects_linear = 0
        count=0
        train_loss = []
        for i, data in enumerate(trainloader):
            count+=1
            inputs, labels = data
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.cuda(), labels.cuda()

            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            output = model(inputs)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)

            # #
            # out_linear= model(inputs)
            # _, linear_preds = torch.max(out_linear.data, 1)
            # loss = criterion(out_linear, labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新cosine学习率
            lr_scheduler.step(epoch + count / iters)

            if i % opt.print_interval == 0 or output.size()[0] < opt.train_batch_size:
                spend_time = time.time() - begin_time
                logger.info(
                    ' Epoch:{}({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        epoch, count, total_iters,
                        loss.item(), optimizer.param_groups[-1]['lr'],
                        spend_time / count * total_iters // 60 - spend_time // 60))
            #
            running_corrects_linear += torch.sum(output == labels.data)
            train_loss.append(loss.item())
            writer.add_scalar('train_loss',loss.item(), global_step=log_train)
            log_train+=1
            #
        #lr_scheduler.step()
        val_acc,val_loss= val_model(model, criterion)
        epoch_acc_linear = running_corrects_linear.double() / total_iters / opt.train_batch_size
        logger.info('valLoss: {:.4f} valAcc: {:.4f}'.format(val_loss,val_acc))
        logger.info('Epoch:[{}/{}] train_acc={:.3f} '.format(epoch, opt.max_epoch,
                                                                    epoch_acc_linear))
        #
        model_out_path = model_save_dir + "/" + '{}_'.format(model_name) + str(epoch) + '.pth'
        best_model_out_path = model_save_dir + "/" + '{}_'.format(model_name) + 'best' + '.pth'
        #model_out_path = '{}_'.format(model_name) + str(epoch) + '.pth'
        #save the best model
        if val_acc > best_score:
            best_score = val_acc
            best_epoch=epoch
            torch.save(model.state_dict(), best_model_out_path)
            logger.info("save best epoch: {} best acc: {}".format(best_epoch,val_acc))
        #save based on epoch interval
        if epoch % opt.save_interval == 0 and epoch>opt.min_save_epoch:
            torch.save(model.state_dict(), model_out_path)
    #
    logger.info('Best acc: {:.3f} Best epoch:{}'.format(best_score,best_epoch))
    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    writer.close()

@torch.no_grad()
def val_model(model, criterion):
    val_dataset = fuDataset(opt.train_val_data, opt.train_label_csv, phase='val', input_size=opt.input_size)
    val_loader = DataLoader(val_dataset,
                             batch_size=opt.val_batch_size,
                             shuffle=False,
                             num_workers=opt.num_workers)
    dset_sizes=len(val_dataset)
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    pres_list=[]
    labels_list=[]
    for data in val_loader:
        inputs, labels = data
        labels = labels.type(torch.LongTensor)
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        if cont == 0:
            outPre = outputs.data.cpu()
            outLabel = labels.data.cpu()
        else:
            outPre = torch.cat((outPre, outputs.data.cpu()), 0)
            outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
        pres_list+=preds.cpu().numpy().tolist()
        labels_list+=labels.data.cpu().numpy().tolist()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        cont += 1
    #
    val_acc = accuracy_score(labels_list, pres_list)
    return val_acc,running_loss / dset_sizes

#
if __name__ == "__main__":
    from config import Config
    from model import EfficientNet
    opt = Config()
    torch.cuda.empty_cache()
    device = torch.device(opt.device)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    model_name=opt.backbone
    model_save_dir =os.path.join(opt.checkpoints_dir , model_name)
    if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
    logger = get_logger(os.path.join(model_save_dir,'log.log'))
    logger.info('Using: {}'.format(model_name))
    logger.info('InputSize: {}'.format(opt.input_size))
    logger.info('optimizer: {}'.format(opt.optimizer))
    logger.info('lr_init: {}'.format(opt.lr))
    logger.info('batch size: {}'.format(opt.train_batch_size))
    logger.info('criterion: {}'.format(opt.loss))
    logger.info('Using label smooth: {}'.format(opt.use_smooth_label))
    logger.info('lr_scheduler: {}'.format(opt.lr_scheduler))
    logger.info('Using the GPU: {}'.format(str(opt.gpu_id)))

    model = EfficientNet.from_pretrained(model_name='efficientnet-b7', num_classes=2)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4 ,weight_decay=5e-4)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-6, last_epoch=-1)
    train_model(model, criterion, optimizer,
              lr_scheduler=lr_scheduler)
