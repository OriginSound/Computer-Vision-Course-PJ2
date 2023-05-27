import torch 
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

import numpy as np 
from aug import Cutmix, Cutout, Mixup


class BaselineDataset(Dataset):
    def __init__(self, train=True):
        self.dataset = torchvision.datasets.CIFAR100(
            "./cifar_data", 
            train=train, 
            transform=transforms.ToTensor(),
            download=True)
        self.length = len(self.dataset)
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        target = torch.zeros(100)
        target[label] = 1
        return image, target
    

class CutOutDataset(Dataset):
    def __init__(self, prob=0.2):
        self.cutout = Cutout()
        self.prob = prob

        self.dataset = torchvision.datasets.CIFAR100(
            "./cifar_data", 
            train=True, 
            transform=transforms.ToTensor(),
            download=True)
        self.length = len(self.dataset)
            
    def __len__(self):
        return self.length 
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        target = torch.zeros(100)
        target[label] = 1

        if np.random.rand() < self.prob:
            image, target = self.cutout((image, target))
        
        return image, target
    

class MixUpDataset(Dataset):
    def __init__(self, prob=0.2):
        self.mixup = Mixup()
        self.prob = prob

        self.dataset = torchvision.datasets.CIFAR100(
            "./cifar_data", 
            train=True, 
            transform=transforms.ToTensor(),
            download=True)
        self.length = len(self.dataset)
            
    def __len__(self):
        return self.length 
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        target = torch.zeros(100)
        target[label] = 1

        if np.random.rand() < self.prob:
            image_, label_ = self.dataset[np.random.randint(self.length)]
            target_ = torch.zeros(100)
            target_[label_] = 1
            image, target = self.mixup((image, target), (image_, target_))

        return image, target 


class CutMixDataset(Dataset):
    def __init__(self, prob=0.2):
        self.cutmix = Cutmix()
        self.prob = prob

        self.dataset = torchvision.datasets.CIFAR100(
            "./cifar_data", 
            train=True, 
            transform=transforms.ToTensor(),
            download=True)
        self.length = len(self.dataset)
            
    def __len__(self):
        return self.length 
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        target = torch.zeros(100)
        target[label] = 1

        if np.random.rand() < self.prob:
            image_, label_ = self.dataset[np.random.randint(self.length)]
            target_ = torch.zeros(100)
            target_[label_] = 1
            image, target = self.cutmix((image, target), (image_, target_))

        return image, target 


        