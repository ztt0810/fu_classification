from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
import pandas as pd
import numpy as np
import os

train_label=pd.read_csv('./data/train_label.csv')
len(train_label[train_label['label']==1.0]),len(train_label[train_label['label']==0.0]) #可见正负样本十分均衡

class fuDataset(Dataset):
    def __init__(self, root, train_label_csv, phase='train', input_size=224):
        self.phase = phase
        train_val_label=pd.read_csv('./data/train_label.csv')
        val_ids=[i for i in range(len(train_label)) if i%5==0]#验证集
        train_ids=[i for i in range(len(train_label)) if i%5!=0]#训练集
        if phase=='train':
            img_label=train_val_label[train_val_label.index.isin(train_ids)].reset_index()
            self.img_names=[os.path.join(root,i) for i in img_label['img_id'].values]
            self.labels=img_label['label'].values
        else:
            img_label=train_val_label[train_val_label.index.isin(val_ids)].reset_index()
            self.img_names=[os.path.join(root,i) for i in img_label['img_id'].values]
            self.labels=img_label['label'].values
        #使用全部数据训练（不要验证集）
        self.img_names=[os.path.join(root,i) for i in train_val_label['img_id'].values]
        self.labels=train_val_label['label'].values
        #
        normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        if self.phase == 'train':
            self.transforms = T.Compose([
                T.Resize((input_size,input_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.25),
                T.RandomRotation(degrees=(-20,20)),
                T.ColorJitter(0.2,0.2),
                T.ToTensor(),

                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((input_size,input_size)),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        img_path = self.img_names[index]
        data = Image.open(img_path)
        data = data.convert('RGB')
        data = self.transforms(data)
        label = np.int32(self.labels[index])
        return data.float(), label

    def __len__(self):
        return len(self.img_names)
