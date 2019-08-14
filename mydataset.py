from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pdb
import torch

def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader = default_loader):
        fh = open(txt, 'r')
        # pdb.set_trace()
        imgs = []
        # import pdb; pdb.set_trace()
        for line in fh:
            # 读取一行
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            if len(words)>2:
                words[0] = str((words[0]))+' '+str((words[1]))
                words[1] = words[2]
            # print(len(words))
            # ims:[('./data/wcedata/train/normal/normal.283.jpg', 2), ('./data/wcedata/train/normal/normal.412.jpg', 2)]
            imgs.append((words[0],int(words[1])))
            # print(words[0],int(words[1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.classes = {'bleeding': 0, 'colon': 1, 'normal': 2}
        fh.close()

    def __getitem__(self, index):
        # pdb.set_trace()
        fn, label = self.imgs[index]
        # print("label:"+str(label))
        img = self.loader(fn)
        # img = np.asarray(img)
        # label = np.asarray(label)
        if self.transform is not None:
            img = self.transform(img)
            # label = self.transform(label)
        # label = torch.IntTensor(label)
        return img,label

    def __len__(self):
        return len(self.imgs)