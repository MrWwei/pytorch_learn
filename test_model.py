import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
import torch.optim as optim
from PIL import Image
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

img = default_loader('data/wcedata/val/colon/colon.47.jpg')
img = transforms(img)
img = img.unsqueeze(0)

input = torch.tensor(img)
input.to(device)
model_ft.to(device)
output = model_ft(input)
_, preds = torch.max(output, 1)
print(preds)

# model_ft = model_ft.to(device)

# criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
