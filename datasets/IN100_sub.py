import torchvision
from datasets.base import get_counts
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class IN100_sub(torchvision.datasets.ImageFolder):
    def __init__(self, root, split='train', transform=None, group=0, cfg=None):
        self.group = group
        if split == "train":
            root = root + '/train'
        else:
            root = root + '/val'
        super().__init__(root, transform=transform)
        self.groups = [0]*len(self.samples)
        self.group_names = ['all']
        self.split = split

        self.class_names = self.classes # n03733281
        self.class_map = None
        self.targets = [s[1] for s in self.samples]

        self.class_weights = get_counts(self.targets)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, 0