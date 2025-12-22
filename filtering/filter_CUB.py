import torch
import torchvision
import torchvision.transforms as transforms
from omegaconf import OmegaConf

import os
import sys
sys.path.insert(0, os.getcwd())
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
from utils import nest_dict, read_unknowns, flatten_config
# from filtering.filtering_utils import get_clip_features, get_features, load_checkpoint
# from cleanlab.count import get_confident_thresholds
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
print(f"device: {device}")
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.wandb_silent:
    os.environ['WANDB_SILENT']="true"

if args.test:
    print("-------------------------------------------------------------")
    print("------------------------- TEST MODE -------------------------")
    print("-------------------------------------------------------------")
    run = wandb.init(project="SEI-filter", name='test', group=args.name, config=flatten_config(args))
else:
    print("project=SEI-filter", f"group={args.name}, config={flatten_config(args)}")

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
clip_model, clip_transform = clip.load(args.filter.model, device="cuda", download_root="/grp01/cs_hszhao/cs002u03/ckpt/clip")

# Data
print('==> Preparing data..')
transform = get_train_transform(args.data.base_dataset, model=args.model, augmentation=args.data.augmentation)
clip_dataset = get_aug_dataset(args, clip_transform)
clip_augloader = torch.utils.data.DataLoader(clip_dataset, batch_size=args.data.batch, shuffle=False, num_workers=8)

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

# customized SEI filter
def semantic_filter_SEI(embeddings, labels, negative_text):
    """Filter out images that are not similar to the text prompt"""
    model, preprocess = clip.load("ViT-L/14", device="cuda")
    predictions = []
    remove_idxs = []
    idxs = []
    with torch.no_grad():
        for idx, img_feature in enumerate(embeddings):
            img_feature = img_feature.unsqueeze(0)
            word = words[labels[idx]]
            text = [f"a photo of a {word}"]
            texts = clip.tokenize(text + negative_text).to("cuda")
            text_features = model.encode_text(texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            # print(f"image_feature.shape {img_feature.shape}, text_features.shape: {text_features.shape}")
            similarity = (100.0 * img_feature @ text_features.T).softmax(dim=-1)
            prediction = torch.argmax(similarity, dim=1).cpu().numpy()
            predictions.append(prediction)
            if prediction == 0:
                idxs.append(idx) # kept
            else:
                remove_idxs.append(idx)
                print("=============================================")
                print("prompt: {}".format(f"a photo of a {word}"))
                print("removed img: ", clip_dataset.samples[idx][0])
                
    predictions = torch.tensor(predictions).to("cuda")
    return predictions, remove_idxs, idxs

# customized SEI filter with rank
def semantic_filter_rank(embeddings, labels, words=None):
    """Filter out images that are not similar to the text prompt"""
    model, preprocess = clip.load("ViT-L/14", device="cuda", download_root="/grp01/cs_hszhao/cs002u03/ckpt/clip")
    # predictions = []
    remove_idxs = []
    idxs = []

    category_similarity = {label.item(): [] for label in set(labels)}

    with torch.no_grad():
        for idx, img_feature in enumerate(embeddings):
            img_feature = img_feature.unsqueeze(0)
            word = words[labels[idx]]
            text = [f"a photo of a {word}"]
            texts = clip.tokenize(text).to("cuda")
            text_features = model.encode_text(texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            # print(f"image_feature.shape {img_feature.shape}, text_features.shape: {text_features.shape}")
            similarity = (100.0 * img_feature @ text_features.T)
            print(f"sim shape :{similarity.shape}")
            # prediction = torch.argmax(similarity, dim=1).cpu().numpy()
            # predictions.append(prediction)
            category_similarity[labels[idx].item()].append((similarity[0,0].item(), idx))

    for category, sims in category_similarity.items():
        print(f"Category {category} sims before sorting: {sims[:5]}") 
        sims.sort(reverse=True, key=lambda x: x[0])
        print(f"Category {category} sims before sorting: {sims[:5]}") 
        length_sim = len(sims)
        print(f"sim length: {length_sim}")
        best_half = sims[:length_sim//2]
        worst_half = sims[length_sim//2:]
        
        for _, idx in best_half:
            idxs.append(idx)  # half best
        for _, idx in worst_half:
            remove_idxs.append(idx)  # half worst
            print("prompt: {}".format(f"a photo of a {words[labels[idx]]}"))
            print("removed img: ", clip_dataset.samples[idx][0])
                
    # predictions = torch.tensor(predictions).to("cuda")
    print(f"kept length: {len(idxs)}")
    return remove_idxs, idxs

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
        ax.set_title(clip_dataset.classes[cap], fontsize=20)
        ax.axis('off')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

if not args.filter.load or not os.path.exists(f"{args.data.embedding_root}/{args.data.base_dataset}/{args.name}/aug_data.pt"):
    # compute clip embeddings
    aug_emb, aug_labels, aug_groups = get_clip_features(clip_model, clip_augloader)

    if not args.test:
        if not os.path.exists(f"{args.filter.save_dir}/{args.name}"):
            os.makedirs(f"{args.filter.save_dir}/{args.name}")
            os.makedirs(f"{args.filter.save_dir}/{args.name}/samples")
            os.makedirs(f"{args.filter.save_dir}/{args.name}/filtered_idxs")
        
        if not os.path.exists(f"{args.data.embedding_root}/{args.data.base_dataset}/{args.name}"):
            os.makedirs(f"{args.data.embedding_root}/{args.data.base_dataset}/{args.name}")
        print("Saving predictions...")
        aug_data = {"clip_embeddings": aug_emb, "labels": aug_labels, "groups": aug_groups}
        torch.save(aug_data, f"{args.data.embedding_root}/{args.data.base_dataset}/{args.name}/aug_data.pt")

else:
    aug_data = torch.load(f"{args.data.embedding_root}/{args.data.base_dataset}/{args.name}/aug_data.pt")
    aug_emb, aug_labels = aug_data["clip_embeddings"], aug_data["labels"]

aug_emb = aug_emb.cuda()
words = clip_dataset.class_names
if args.data.base_dataset in ["Flowers102"]:
    print("processing '_' for {}".format(args.data.base_dataset))
    words = [" ".join(w.split("_")[1:]) for w in words]
elif args.data.base_dataset in ["Cub2011"]:
    print("processing '.' for {}".format(args.data.base_dataset))
    words = [" ".join(w.split(".")[1:]) for w in words]
print(f"words: {words}")

# aug_labels, base_labels = [int(s[1]) for s in dataset.samples], [int(s[1]) for s in trainset.samples]

##########################################################################################################
################################ Find semantic errors ####################################################
##########################################################################################################
# predictions, semantic_filtered, kept_idxs = semantic_filter_SEI(aug_emb, aug_labels, list(args.filter.negative_prompts))
semantic_filtered, kept_idxs = semantic_filter_rank(aug_emb, aug_labels, words=words)


print(f"semantic filter removed = {len(semantic_filtered)}")
print(f"semantic filter kept = {len(kept_idxs)}")
print(f"semantic filter removed ratio = {len(semantic_filtered) / len(aug_emb)}")
np.save(f"{args.filter.save_dir}/{args.name}/filtered_idxs/semantic_filtered.npy", semantic_filtered)
np.save(f"{args.filter.save_dir}/{args.name}/filtered_idxs/semantic_kept.npy", kept_idxs)

# filtered = np.unique(np.concatenate((semantic_filtered, conf_incorrect_idxs, conf_correct_idxs)))
filtered = semantic_filtered
kept = np.array([i for i in range(len(aug_emb)) if i not in filtered])
np.save(f"{args.filter.save_dir}/{args.name}/filtered_idxs/filtered.npy", filtered)
np.save(f"{args.filter.save_dir}/{args.name}/filtered_idxs/kept.npy", kept)
print(f"Total filter removed {len(filtered)}/{len(aug_emb)} images")
print(f"total filter removed = {len(filtered)}")
print(f"total filter removed ratio = {len(filtered) / len(aug_emb)}")

aug_labels = np.array(aug_labels)

# get num filtered by class counts
semantic_filtered_counts = collections.Counter([l for i, l in enumerate(aug_labels) if i in semantic_filtered])
# easy_filtered_counts = collections.Counter([l for i, l in enumerate(aug_labels) if i in conf_correct_idxs])
# mislabled_filtered_counts = collections.Counter([l for i, l in enumerate(aug_labels) if i in conf_incorrect_idxs])
# filtered_counts = collections.Counter([l for i, l in enumerate(aug_labels) if i in filtered])
# put into dataframe of label, semantic_filtered, nn_filtered, filtered, total
counts = []
for l in range(len(clip_dataset.classes)):
    counts.append([clip_dataset.classes[l], semantic_filtered_counts[l], len(aug_labels[aug_labels == l])])
counts = pd.DataFrame(counts, columns=["label", "semantic_filtered", "total"])
counts.to_csv(f"{args.filter.save_dir}/{args.name}/label_predictions.csv")
# label_table = wandb.Table(dataframe=counts[counts['filtered'] > 0])
# print({"Label predictions": label_table})

edit_filenames = [clip_dataset.samples[i][0] for i in range(len(clip_dataset))]
# plot random sample of filtered vs original dataset
filtered_sample = np.random.choice(filtered, 10)
print(len(edit_filenames), filtered_sample)
# yet another hack, it will throw an error if you dont filter out any examples from a class
try:
    filtered_vis = [Image.open(edit_filenames[i]) for i in filtered_sample]
    filtered_captions = [aug_labels[i] for i in filtered_sample]
    unfiltered_sample = np.random.choice(kept, 10)
    unfiltered_vis = [Image.open(edit_filenames[i]) for i in unfiltered_sample]
    unfiltered_captions = [aug_labels[i] for i in unfiltered_sample]
    plot_imgs(filtered_vis+unfiltered_vis, filtered_captions+unfiltered_captions, n_rows=2, save_path=f"{args.filter.save_dir}/{args.name}/samples/filtered_vs_unfiltered.png")
except:
    pass