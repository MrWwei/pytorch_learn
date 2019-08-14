# Author: Wentao WEI
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
import torch.optim as optim
from PIL import Image
import pdb
import numpy as np
'''
pytorch单张或者多张图片分类测试
'''
def default_loader(path):
    return Image.open(path).convert('RGB')

# 描述模型
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, 3)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载权重参数
model_ft.load_state_dict(torch.load('./model_25.pth'))
# eval模式与train不同点：eval会取消bn和dropout
model_ft.eval()

# 对图片进行转化处理
transforms=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
# 单张图片
path1 = 'data/wcedata/val/colon/colon.47.jpg'
img1 = default_loader(path1)
img1 = transforms(img1)
img1 = img1.unsqueeze(0)

path2 = 'data/wcedata/val/bleeding/bleeding.47.jpg'
img2 = default_loader(path2)
img2 = transforms(img2)
img2 = img2.unsqueeze(0)

imgs = np.concatenate((img1,img2))

labels = []
paths = [path1,path2]
for path in paths:
    if path.split('/')[-2] == 'bleeding':
        labels.append(0) 
    elif path.split('/')[-2] == 'colon':
        labels.append(1)
    else:
        labels.append(2)



print(labels)
# numpy转成Tensor
input = torch.tensor(imgs)

# 多张图片
# pdb.set_trace()
# 如果直接input.to(device)会报错：‘Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same’
input = input.to(device)
model_ft.to(device)
output = model_ft(input)
_, preds = torch.max(output, 1)
# print(preds)
for idx,pred in enumerate(preds):
    if int(pred) == labels[idx]:
        print('right')
    else:
        print('error')


