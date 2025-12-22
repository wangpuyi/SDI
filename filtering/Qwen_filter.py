from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import random
import argparse
from omegaconf import OmegaConf
from utils import nest_dict, read_unknowns, flatten_config
import os
import wandb
import numpy as np
import datasets
from datasets.base import CombinedDataset
from helpers.load_dataset import get_train_transform, get_dataset
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
from tqdm import tqdm

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

def Qwenfilter5to1(text):
    mapping = {
        'first': 0,
        'second': 1,
        'third': 2,
        'fourth': 3,
        'fifth': 4
    }
    
    for word, number in mapping.items():
        if word in text:
            return number
    
    return random.randint(0, 4)


parser = argparse.ArgumentParser(description='parameter loading')
parser.add_argument('--config', default='configs/base.yaml', help="config file")
parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values ")
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
random.seed(args.seed)

def get_aug_dataset(args, transform):
    if type(args.data.extra_root) != str:
        print("Roots", list(args.data.extra_root))
        roots, dsets = list(args.data.extra_root), []
        for i, r in enumerate(roots):
            dsets.append(getattr(datasets.base, args.data.extra_dataset)(args.data.extra_root, transform=transform, cfg=args))
        dataset = CombinedDataset(dsets)
    else:
        dataset = getattr(datasets.base, args.data.extra_dataset)(args.data.extra_root, transform=transform, cfg=args)
        print("DATASET SIZE", len(dataset))
    return dataset

print('==> Preparing data...')
transform = get_train_transform(args.data.base_dataset, model=args.model, augmentation=args.data.augmentation)
dataset = get_aug_dataset(args, transform)
augloader = torch.utils.data.DataLoader(dataset, batch_size=args.data.batch, shuffle=False, num_workers=8)

def Qwen_filter(args, dataset):
    '''
    input: dataset
    output: removed_idxs, idxs
    '''
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    csv = pd.read_csv(args.sentences_csv)
    disorder_sentences = csv['sentence'].tolist()
    index = csv['index'].tolist()
    index = [int(i) for i in index]
    sorted_pairs = sorted(zip(index, disorder_sentences))
    sentences = [sentence for _, sentence in sorted_pairs]

    def Qwenfilter5to1(text):
        mapping = {
            'first': 0,
            'second': 1,
            'third': 2,
            'fourth': 3,
            'fifth': 4
        }
        
        for word, number in mapping.items():
            if word in text.lower():
                return number
        
        return random.randint(0, 4)
    
    idxs, removed_idxs = [], []
    print(f'length of dataset: {len(dataset)}')
    for i in tqdm(range(len(dataset)//5)):
        sentence = sentences[i]
        query = tokenizer.from_list_format([
            {'image': f'/mnt/hwfile/gveval/yangshuo/dataset/MoMA/CUB2011/sample_act3_withClass_5to1/{i}_concatenated.jpg'}, # Either a local path or an url
            {'text': f"The image provided is a collection of five images. Identify which image from the left most accurately depicts {sentence} You should explicitly reply the index of the chosen image and provide a rationale for your choice. In your assessment, prioritize the species and shape of the bird over its action."},
        ])
        response, history = model.chat(tokenizer, query=query, history=None)
        prediction = Qwenfilter5to1(response)
        assert type(prediction) == int, 'prediction is not an integer'
        # print(response)
        # print(prediction) 
        idxs.append(i*5 + prediction)
        removed_idxs += [5*i+p for p in range(5) if p != prediction]

    return removed_idxs, idxs

def plot_imgs(images, captions, n_rows=1, save_path=None):
    n_cols = len(images) //n_rows
    fig = plt.figure(figsize=(5*n_cols, 5*n_rows))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.25)

    for ax, im, cap in zip(grid, images, captions):
        ax.imshow(im.resize((256, 256)))
        ax.set_title(dataset.classes[cap], fontsize=20)
        ax.axis('off')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

##############################################################################################################
################################ Qwen filter ####################################################
##############################################################################################################
if not os.path.exists(f"{args.filter.save_dir}/{args.name}"):
    os.makedirs(f"{args.filter.save_dir}/{args.name}")
    os.makedirs(f"{args.filter.save_dir}/{args.name}/samples")
    os.makedirs(f"{args.filter.save_dir}/{args.name}/filtered_idxs")

aug_labels, aug_img_names = [int(s[1]) for s in dataset.samples], [item[0].split('/')[-1] for item in dataset.samples]
Qwen_filtered, kept_idxs = Qwen_filter(args, dataset)

wandb.summary["Qwen filter removed"] = len(Qwen_filtered)
wandb.summary["Qwen filter kept"] = len(kept_idxs)
wandb.summary["Qwen filter removed ratio"] = len(Qwen_filtered) / len(aug_labels)

np.save(f"{args.filter.save_dir}/{args.name}/filtered_idxs/Qwen_filtered.npy", Qwen_filtered)
np.save(f"{args.filter.save_dir}/{args.name}/filtered_idxs/Qwen_kept.npy", kept_idxs)


filtered = Qwen_filtered
print(f"Total filter removed {len(filtered)}/{(len(Qwen_filtered)+len(kept_idxs))} images")
wandb.summary["total filter removed"] = len(filtered)
wandb.summary["total filter removed ratio"] = len(filtered) / len(aug_labels)

aug_labels = np.array(aug_labels)


kept = np.array([i for i in range(len(aug_labels)) if i not in filtered])
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
