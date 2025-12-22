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

def load_checkpoint(args, net):
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if args.filter.checkpoint_name:
        checkpoint_name = f'./checkpoint/{args.filter.checkpoint_name}'
    checkpoint = torch.load(checkpoint_name)

    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['net'].items():
        if 'module' not in k:
            k = 'module.'+k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k]=v
    
    print(f"Loaded checkpoint at epoch {checkpoint['epoch']} from {checkpoint_name}")
    # net.load_state_dict(checkpoint['net'])
    net.load_state_dict(new_state_dict)

    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    return net, best_acc, start_epoch

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
if 'Extra' in args.data.base_dataset: 
    trainset, valset, testset, dataset = get_dataset(args.data.base_dataset, transform=transform, val_transform=transform, root=args.data.base_root)
else: # default Cub2011
    dataset = get_aug_dataset(args, transform) # extra data
    clip_dataset = get_aug_dataset(args, clip_transform)
    trainset, valset, testset, _ = get_dataset(args.data.base_dataset, transform=transform, val_transform=transform, root=args.data.base_root)
    clip_trainset, clip_valset, clip_testset, _ = get_dataset(args.data.base_dataset, transform=clip_transform, val_transform=transform, root=args.data.base_root)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.data.batch, shuffle=False, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.data.batch, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.data.batch, shuffle=False, num_workers=2)
augloader = torch.utils.data.DataLoader(dataset, batch_size=args.data.batch, shuffle=False, num_workers=2)

clip_trainloader = torch.utils.data.DataLoader(clip_trainset, batch_size=args.data.batch, shuffle=False, num_workers=2)
clip_valloader = torch.utils.data.DataLoader(clip_valset, batch_size=args.data.batch, shuffle=False, num_workers=2)
clip_testloader = torch.utils.data.DataLoader(clip_testset, batch_size=args.data.batch, shuffle=False, num_workers=2)
clip_augloader = torch.utils.data.DataLoader(clip_dataset, batch_size=args.data.batch, shuffle=False, num_workers=2)

def get_clip_features(model, loader):
    model.eval()
    all_features = []
    all_labels = []
    all_groups = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            images, labels, groups = batch
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)
            all_groups.append(groups)

    return torch.cat(all_features).cpu(), torch.cat(all_labels).cpu(), torch.cat(all_groups).cpu()

# get cosine similarity per class between dataset and base
def get_cosine_similarity(embeddings, labels, base_embeddings, base_labels):
    # get cosine similarity per class between dataset and base
    # normalize
    embeddings /= embeddings.norm(dim=-1, keepdim=True)
    base_embeddings /= base_embeddings.norm(dim=-1, keepdim=True)
    cosine_sim = []
    for i in np.unique(labels):
        class_embeddings = embeddings[labels == i]
        class_base_embeddings = base_embeddings[base_labels == i]
        # get cos sim to nn in training set
        cosine_sim_cls = []
        for emb in class_embeddings:
            cos_sim = torch.unsqueeze(emb, 0) @ class_base_embeddings.T
            cosine_sim_cls.append(torch.max(cos_sim, dim=1)[0])
        cosine_sim.append(torch.stack(cosine_sim_cls))
    return cosine_sim

def semantic_filter(dataset, text, negative_text = ["a photo of an object", "a photo of a scene", "a photo of geometric shapes", "a photo", "an image", "a black photo"], threshold=0.9):
    """Filter out images that are not similar to the text prompt"""
    model, preprocess = clip.load("ViT-L/14", device="cuda")
    text = [text] if type(text) == str else text
    texts = clip.tokenize(text + negative_text).to("cuda") # list of text
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.data.batch, shuffle=False, num_workers=4)
    text_features = model.encode_text(texts)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    ret = []
    embeddings = []
    with torch.no_grad():
        for images, labels, _, _ in tqdm(loader):
            image_features = model.encode_image(images.cuda())
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embeddings += [image_features]
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            ret.append(similarity)
    results = torch.cat(ret)
    predictions = torch.argmax(results, dim=1).cpu().numpy()
    idxs = [p for p in range(len(predictions)) if predictions[p] in list(range(len(text)))]
    remove_idxs = [p for p in range(len(predictions)) if predictions[p] not in list(range(len(text)))]
    return predictions, remove_idxs, idxs, torch.cat(embeddings)

def semantic_filter_saved(embeddings, text, negative_text):
    """Filter out images that are not similar to the text prompt"""
    model, preprocess = clip.load("ViT-L/14", device="cuda")
    text = [text] if type(text) == str else text
    texts = clip.tokenize(text + negative_text).to("cuda") # [number of input strings, context_length]
    text_features = model.encode_text(texts)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        image_features = embeddings # CLIP encoded
        print("image_features.shape, text_features.shape: ", image_features.shape, text_features.shape)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        predictions = torch.argmax(similarity, dim=1).cpu().numpy()
        idxs = [p for p in range(len(predictions)) if predictions[p] in list(range(len(text)))]
        remove_idxs = [p for p in range(len(predictions)) if predictions[p] not in list(range(len(text)))]
    return predictions, remove_idxs, idxs

def semantic_filter_1out5(args, embeddings, img_names):
    """Filter out images that are not similar to the text prompt"""
    model, preprocess = clip.load("ViT-L/14", device="cuda")
    csv = pd.read_csv(args.sentences_csv)
    disorder_prompts = csv['sentence'].tolist()
    index = csv['index'].tolist()
    index = [int(i) for i in index]
    sorted_pairs = sorted(zip(index, disorder_prompts))
    prompts = [prompt for _, prompt in sorted_pairs]

    idxs, remove_idxs = [], []
    print(f'length of embedding: {len(embeddings)}')
    for i in range(0,len(embeddings)//5):
        print(f'i: {i}')
        prompt_index = int(img_names[i].split('-')[0])

        with torch.no_grad():
            text = [prompts[prompt_index]]
            text = clip.tokenize(text).to("cuda")
            # [number of input strings, context_length]
            text_feature = model.encode_text(text)
            text_feature /= text_feature.norm(dim=-1, keepdim=True)
            embeddings_batch = embeddings[5*i:5*(i+1)]
            # print(f'len of embed batch: {len(embeddings_batch)}')
            print("image_features.shape, text_features.shape: ", embeddings_batch.shape, text_feature.shape)
            similarity = (100.0 * embeddings_batch @ text_feature.T).softmax(dim=0)
            predictions = torch.argmax(similarity, dim=0).cpu().numpy()
            # print(f'predictions: {predictions}') # numpy array
            # print(f'pred size: {predictions.shape}')
            idxs.append(5*i+predictions[0])
            remove_idxs += [5*i+p for p in range(5) if p != predictions[0]]
    print(f'idxs: {idxs}')
    return predictions, remove_idxs, idxs


def plot_imgs(images, captions, n_rows=1, save_path=None):
    n_cols = len(images) // n_rows
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(n_rows, n_cols),  # creates 2x2 grid of axes
                    axes_pad=0.25,  # pad between axes in inch.
                    )

    for ax, im, cap in zip(grid, images, captions):
        # Iterating over the grid returns the Axes.
        ax.imshow(im.resize((224, 224)))
        ax.set_title(dataset.classes[cap], fontsize=20)
        ax.axis('off')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()



if not args.filter.load or not os.path.exists(f"{args.data.embedding_root}/{args.data.base_dataset}/{args.name}/train_data.pt"):
    # compute clip embeddings
    train_emb, train_labels, train_groups = get_clip_features(clip_model, clip_trainloader)
    aug_emb, aug_labels, aug_groups = get_clip_features(clip_model, clip_augloader)
    val_emb, val_labels, val_groups = get_clip_features(clip_model, clip_valloader)
    test_emb, test_labels, test_groups = get_clip_features(clip_model, clip_testloader)

    aug_img_names = [item[0].split('/')[-1] for item in dataset.samples]

    if not args.test:
        if not os.path.exists(f"{args.filter.save_dir}/{args.name}"):
            os.makedirs(f"{args.filter.save_dir}/{args.name}")
            os.makedirs(f"{args.filter.save_dir}/{args.name}/samples")
            os.makedirs(f"{args.filter.save_dir}/{args.name}/filtered_idxs")
        
        if not os.path.exists(f"{args.data.embedding_root}/{args.data.base_dataset}/{args.name}"):
            os.makedirs(f"{args.data.embedding_root}/{args.data.base_dataset}/{args.name}")
        print("Saving predictions...")
        train_data = {"clip_embeddings": train_emb, "labels": train_labels , "groups": train_groups}
        aug_data = {"clip_embeddings": aug_emb, "labels": aug_labels, "names": aug_img_names, "groups": aug_groups}
        val_data = {"clip_embeddings": val_emb, "labels": val_labels, "groups": val_groups}
        test_data = {"clip_embeddings": test_emb, "labels": test_labels, "groups": test_groups}
        torch.save(train_data, f"{args.data.embedding_root}/{args.data.base_dataset}/train_data.pt")
        torch.save(aug_data, f"{args.data.embedding_root}/{args.data.base_dataset}/{args.name}/train_data.pt")
        torch.save(val_data, f"{args.data.embedding_root}/{args.data.base_dataset}/val_data.pt")
        torch.save(test_data, f"{args.data.embedding_root}/{args.data.base_dataset}/test_data.pt")
else:
    train_data = torch.load(f"{args.data.embedding_root}/{args.data.base_dataset}/train_data.pt")
    aug_data = torch.load(f"{args.data.embedding_root}/{args.data.base_dataset}/{args.name}/train_data.pt")
    val_data = torch.load(f"{args.data.embedding_root}/{args.data.base_dataset}/val_data.pt")
    test_data = torch.load(f"{args.data.embedding_root}/{args.data.base_dataset}/test_data.pt")
    train_emb, train_labels = train_data["clip_embeddings"], train_data["labels"]
    aug_emb, aug_labels, aug_img_names  = aug_data["clip_embeddings"], aug_data["labels"], aug_data["names"]
    val_emb, val_labels = val_data["clip_embeddings"], val_data["labels"]
    test_emb, test_labels = test_data["clip_embeddings"], test_data["labels"]

train_emb, aug_emb, val_emb, test_emb = train_emb.cuda(), aug_emb.cuda(), val_emb.cuda(), test_emb.cuda()

aug_labels, base_labels = [int(s[1]) for s in dataset.samples], [int(s[1]) for s in trainset.samples]

##############################################################################################################
################################ 1. Find semantic errors ####################################################
##############################################################################################################

predictions, semantic_filtered, kept_idxs = semantic_filter_1out5(args, aug_emb, aug_img_names)
# base_predictions, base_idxs, _ = semantic_filter_saved(train_emb, args.filter.prompt, list(args.filter.negative_prompts))
wandb.summary["semantic filter removed"] = len(semantic_filtered)
wandb.summary["semantic filter kept"] = len(kept_idxs)
wandb.summary["semantic filter removed ratio"] = len(semantic_filtered) / len(aug_emb)
# wandb.summary['semantic filter removed original'] = len(base_idxs) # filtered
# wandb.summary["semantic filter removed ratio original"] = len(base_idxs) / len(train_emb)
np.save(f"{args.filter.save_dir}/{args.name}/filtered_idxs/semantic_filtered.npy", semantic_filtered)
np.save(f"{args.filter.save_dir}/{args.name}/filtered_idxs/semantic_kept.npy", kept_idxs)

filtered = semantic_filtered
print(f"Total filter removed {len(filtered)}/{len(aug_emb)} images")
wandb.summary["total filter removed"] = len(filtered)
wandb.summary["total filter removed ratio"] = len(filtered) / len(aug_emb)

aug_labels = np.array(aug_labels)


kept = np.array([i for i in range(len(aug_emb)) if i not in filtered])
np.save(f"{args.filter.save_dir}/{args.name}/filtered_idxs/kept.npy", kept)

edit_filenames = [dataset.samples[i][0] for i in range(len(dataset))]
# plot random sample of filtered vs original dataset
filtered_sample = np.random.choice(filtered, 10)
print(len(edit_filenames), filtered_sample)
# yet another hack, it will throw an error if you dont filter out any examples from a class
try:
    filtered_vis = [Image.open(edit_filenames[i]) for i in filtered_sample]
    filtered_captions = [(aug_labels[i], str(int(aug_img_names[i]))) for i in filtered_sample]
    unfiltered_sample = np.random.choice(kept, 10)
    unfiltered_vis = [Image.open(edit_filenames[i]) for i in unfiltered_sample]
    unfiltered_captions = [(aug_labels[i], str(int(aug_img_names[i]))) for i in unfiltered_sample]
    plot_imgs(filtered_vis+unfiltered_vis, filtered_captions+unfiltered_captions, n_rows=2, save_path=f"{args.filter.save_dir}/{args.name}/samples/filtered_vs_unfiltered.png")
except:
    pass
