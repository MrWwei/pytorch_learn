from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pdb
def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader = default_loader):
        fh = open(txt, 'r')
        imgs = []
        # import pdb; pdb.set_trace()
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            if len(words)>2:
                words[0] = str((words[0]))+' '+str((words[1]))
                words[1] = words[2]
            # print(len(words))

            imgs.append((words[0],int(words[1])))
            # print(words[0],int(words[1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
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
        return img,label

    def __len__(self):
        return len(self.imgs)