import torch
import torchvision
import torchvision.transforms as transforms
from omegaconf import OmegaConf

import os
import argparse
import wandb
import clip
import numpy as np
import collections 
import random
from tqdm import tqdm
from PIL import Image
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import datasets
import models
from utils import nest_dict, read_unknowns, flatten_config
# from filtering.filtering_utils import get_clip_features, get_features, load_checkpoint
from cleanlab.count import get_confident_thresholds
from datasets.base import CombinedDataset
from helpers.load_dataset import get_train_transform, get_dataset

parser = argparse.ArgumentParser(description='Dataset Understanding')
parser.add_argument('--config', default='configs/base.yaml', help="config file")
parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values "
                                                "(use dots for.nested=overrides)")
flags, unknown = parser.parse_known_args()

overrides = OmegaConf.from_cli(flags.overrides)
cfg       = OmegaConf.load(flags.config)
base      = OmegaConf.load('configs/base.yaml')
args      = OmegaConf.merge(base, cfg, overrides)
if len(unknown) > 0:
    print(unknown)
    config = nest_dict(read_unknowns(unknown))
    to_merge = OmegaConf.create(config)
    args = OmegaConf.merge(args, to_merge)
args.yaml = flags.config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.wandb_silent:
    os.environ['WANDB_SILENT']="true"

if args.test:
    print("-------------------------------------------------------------")
    print("------------------------- TEST MODE -------------------------")
    print("-------------------------------------------------------------")
    run = wandb.init(project="ALIA-filter", name='test', group=args.name, config=flatten_config(args))
else:
    run = wandb.init(project="ALIA-filter", group=args.name, config=flatten_config(args))

torch.manual_seed(args.seed)
np.random.seed(args.seed)


def get_aug_dataset(args, transform):
    if type(args.data.extra_root) != str:
        print("Roots", list(args.data.extra_root))
        roots, dsets = list(args.data.extra_root), []
        for i, r in enumerate(roots):
            dsets.append(getattr(datasets.base, args.data.extra_dataset)(r, transform=transform, cfg=args, group=i))
        dataset = CombinedDataset(dsets)
    else:
        dataset = getattr(datasets.base, args.data.extra_dataset)(args.data.extra_root, transform=transform, cfg=args)
        print("DATASET SIZE", len(dataset))
    return dataset


# load clip model
clip_model, clip_transform = clip.load(args.filter.model, device="cuda")

# Data
print('==> Preparing data..')
transform = get_train_transform(args.data.base_dataset, model=args.model, augmentation=args.data.augmentation)
if 'Extra' in args.data.base_dataset: # default Cub2011
    trainset, valset, testset, dataset = get_dataset(args.data.base_dataset, transform=transform, val_transform=transform, root=args.data.base_root)
else:
    dataset = get_aug_dataset(args, transform) # extra data
    clip_dataset = get_aug_dataset(args, clip_transform)
    trainset, valset, testset, _ = get_dataset(args.data.base_dataset, transform=transform, val_transform=transform, root=args.data.base_root)
    clip_trainset, clip_valset, clip_testset, _ = get_dataset(args.data.base_dataset, transform=clip_transform, val_transform=transform, root=args.data.base_root)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.data.batch, shuffle=False, num_workers=8)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.data.batch, shuffle=False, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.data.batch, shuffle=False, num_workers=8)
augloader = torch.utils.data.DataLoader(dataset, batch_size=args.data.batch, shuffle=False, num_workers=8)

clip_trainloader = torch.utils.data.DataLoader(clip_trainset, batch_size=args.data.batch, shuffle=False, num_workers=8)
clip_valloader = torch.utils.data.DataLoader(clip_valset, batch_size=args.data.batch, shuffle=False, num_workers=8)
clip_testloader = torch.utils.data.DataLoader(clip_testset, batch_size=args.data.batch, shuffle=False, num_workers=8)
clip_augloader = torch.utils.data.DataLoader(clip_dataset, batch_size=args.data.batch, shuffle=False, num_workers=8)

# print(clip_dataset[:10])
print(len(clip_dataset.samples))
print(dataset.samples[:20])
aug_img_names = [item[0].split('/')[-1] for item in dataset.samples]
print(aug_img_names[:20])