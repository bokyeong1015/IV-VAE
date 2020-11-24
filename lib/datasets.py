import os
import torch

import numpy as np
from torch.utils.data import Dataset

class shapes_im64(Dataset):

    def __init__(self, data_flag=None):

        if data_flag is None:
            raise(RuntimeError("data type should be specified: train/valid/test"))
        else:
            self.data_flag = data_flag

        data = np.load('data/dspirites_im64_' + self.data_flag + '.npz')
        label = np.load('data/dspirites_im64_' + self.data_flag + '_label.npz')

        self.imgs = torch.from_numpy(data['x']).float()
        self.labels = torch.from_numpy(label['y']).long()
        self.labels = self.labels[:, 1]

        self.n_class = self.labels.unique().size(0)

        del data, label

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, index):
        x = self.imgs[index].view(1, 64, 64)
        y = self.labels[index]

        return x, y


class mnist_fashMni_im32(Dataset):

    def __init__(self, dataset=None, data_flag=None):

        if data_flag is None or dataset is None:
            raise(RuntimeError("data should be specified"))
        else:
            self.data_flag = data_flag

        data = torch.load('data/' + dataset + '_im32_' + data_flag + '.pth')

        self.imgs = data['x']
        self.labels = data['y']

        del data

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, index):

        x = self.imgs[index]
        y = self.labels[index]

        return x, y