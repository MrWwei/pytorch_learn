import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
import torch.optim as optim
from PIL import Image
import pdb
import numpy as np
'''
pytorch单张图片分类
'''
def default_loader(path):
    return Image.open(path).convert('RGB')

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, 3)
device = torch.device("cpu")

model_ft.load_state_dict(torch.load('./model_25.pth'))
model_ft.eval()

transforms=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
# 单张图片
img = default_loader('data/wcedata/val/colon/colon.47.jpg')
img = transforms(img)
img = img.unsqueeze(0)

img1 = default_loader('data/wcedata/val/bleeding/bleeding.47.jpg')
img1 = transforms(img1)
img1 = img1.unsqueeze(0)

imgs = np.concatenate((img,img1))

# numpy转成Tensor
input = torch.tensor(imgs)

# 多张图片
# pdb.set_trace()
input.to(device)
model_ft.to(device)
output = model_ft(input)
_, preds = torch.max(output, 1)
print(preds)

