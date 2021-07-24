import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from random import random

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class mydataset(Dataset):
    def __init__(self,root):
        self.classes = os.listdir(root)
        tmp = list(range(len(self.classes)))
        self.cls_idx = dict(zip(self.classes,tmp))
        self.data = []
        self.label = []
        for i in self.classes:
            tmp = os.listdir(root+'\\'+i)
            tmp = [root + '\\' + i + '\\' + file for file in tmp]
            lens = len(tmp)
            self.data.extend(tmp)
            tmp = list(np.ones(lens)*self.cls_idx[i])
            self.label.extend(tmp)

    def __getitem__(self, item):
        data = np.load(self.data[item])
        if data.shape[2] > 100:
            data = data[:,:,0:100,:,:]
            label = self.label[item]
        else:
            item = int(random()*len(self.data))
            label, data = self.__getitem__(item)
        return label, data


    def __len__(self):
        return len(self.data)



