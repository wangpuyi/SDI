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
from PIL import Image, ImageDraw, ImageFont
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

train_data = torch.load(f"{args.data.embedding_root}/{args.data.base_dataset}/train_data.pt")
aug_data = torch.load(f"{args.data.embedding_root}/{args.data.base_dataset}/{args.name}/train_data.pt")
val_data = torch.load(f"{args.data.embedding_root}/{args.data.base_dataset}/val_data.pt")
test_data = torch.load(f"{args.data.embedding_root}/{args.data.base_dataset}/test_data.pt")
train_emb, train_labels, train_logits, train_features = train_data["clip_embeddings"], train_data["labels"], train_data["logits"], train_data["features"]
aug_emb, aug_labels, logits, features = aug_data["clip_embeddings"], aug_data["labels"], aug_data["logits"], aug_data["features"]
val_emb, val_labels, val_logits, val_features = val_data["clip_embeddings"], val_data["labels"], val_data["logits"], val_data["features"]
test_emb, test_labels, test_logits, test_features = test_data["clip_embeddings"], test_data["labels"], test_data["logits"], test_data["features"]

train_emb, aug_emb, val_emb, test_emb = train_emb.cuda(), aug_emb.cuda(), val_emb.cuda(), test_emb.cuda()

##############################################################################################################
################################ 1. Find label errors #######################################################
##############################################################################################################
def get_pred_and_conf(data):
    logits = data['logits'] # (sample_len, num_classes)
    pred = logits.argmax(dim=1)
    return pred.cpu().numpy(), logits.max(dim=1)[0].cpu().numpy()

preds, conf = get_pred_and_conf(train_data)
# train_data['labels'] (sample_len, )
conf_thresh = get_confident_thresholds(train_data['labels'].numpy(), train_data['logits'].cpu().numpy())
# print('conf_thresh', conf_thresh) # 普遍大于0.98 全部大于0.96

for pred, logit_ in zip(preds, train_logits):
    if pred == 0:
        # print(f'pred: 0  logit: {logit_}') # 0.985 其它至少小两个量级 非常confident
        break

# print('logits: ', train_logits.shape) # torch.Size([4994, 200])

aug_preds, aug_conf = get_pred_and_conf(aug_data)

# print('aug_conf: ', aug_conf) # 0.2 0.3 0.4 0.7 都有，极少大于0.98
# both conf_correct_idxs and conf_incorrect_idxs will be filtered out 
correct_idxs = np.where((aug_preds == aug_data['labels'].numpy()))[0]
incorrect_idxs = np.where(aug_preds != aug_data['labels'].numpy())[0]
print('correct_idxs: ', correct_idxs, 'len: ', len(correct_idxs))
print('incorrect_idxs: ', incorrect_idxs, 'len: ', len(incorrect_idxs)) # 正确559个，错误4435个

print('conf_correct: ', aug_conf[correct_idxs]) # 0.1 - 0.9 都有
print('conf_incorrect: ', aug_conf[incorrect_idxs]) # 同样 0.1 - 0.9 都有

# load clip model
clip_model, clip_transform = clip.load(args.filter.model, device="cuda")
clip_dataset = get_aug_dataset(args, clip_transform)

assert clip_dataset.targets == aug_labels.numpy().tolist(), 'the order is different'

# for i in range(len(aug_labels)):
#     img_path = clip_dataset.samples[i][0]
#     img = Image.open(img_path)
#     GT = clip_dataset.targets[i]
#     GT_conf = logits[i][GT]

#     pred = aug_preds[i]
#     pred_conf = logits[i][pred]
#     output_path = f'/mnt/hwfile/gveval/yangshuo/dataset/CUB_conf/{i}.png'

#     text_GT = f"GT: {GT}, GT_conf: {GT_conf:.2f}"
#     text_pred = f"pred: {pred}, pred_conf: {pred_conf:.2f}"
    
#     draw = ImageDraw.Draw(img)
#     font = ImageFont.truetype('/mnt/hwfile/gveval/yangshuo/font/Roboto/Roboto-Black.ttf', size=20)
#     x, y = 10, 10
#     draw.text((x, y), text_GT, font=font, fill=(255, 0, 0))
#     draw.text((x, y+25), text_pred, font=font, fill=(255, 0, 0))

#     img.save(output_path)

for i in correct_idxs:
    print(f'/mnt/hwfile/gveval/yangshuo/dataset/CUB_conf/{i}.png')

