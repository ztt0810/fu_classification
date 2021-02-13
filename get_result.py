import os
import pandas as pd
import torch
from model import EfficientNet
from torchvision import transforms as T
from PIL import Image
from config import Config
from pytorch_toolbelt.inference import tta


submit=pd.read_csv('./data/result.csv',header=None)
submit.columns=['name']
model = EfficientNet.from_pretrained(model_name='efficientnet-b7', num_classes=2)
net_weight='./ckpt/efficientnet/efficientnet_30.pth'
model.load_state_dict(torch.load(net_weight))
model = model.cuda()
model.eval()
opt = Config()
#
infer_transforms=T.Compose([
                T.Resize((opt.input_size,opt.input_size)),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
result=[]
test_dir='./data/test/'
for name in submit['name'].values:
    img_path=os.path.join(test_dir,name)
    data = Image.open(img_path)
    data = data.convert('RGB')
    data = infer_transforms(data)
    data=data.unsqueeze(0)  
    inputs= data.cuda()
    with torch.no_grad():
        outputs = tta.fliplr_image2label(model, inputs)
    _, preds = torch.max(outputs.data, 1)
    result.append(preds.cpu().data.numpy()[0])
    #
submit['label']=result
submit.to_csv('submit.csv',index=False,header=None)