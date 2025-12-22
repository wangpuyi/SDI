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

IN200_wnids = ['n01514668', 'n01530575', 'n01616318', 'n01629819', 'n01687978', 
         'n01695060', 'n01698640', 'n01729977', 'n01773157', 'n01773549',
         'n01784675', 'n01806143', 'n01824575', 'n01829413', 'n01860187',
         'n01872401', 'n01877812', 'n01883070', 'n01924916', 'n01943899', 
         'n01980166', 'n02002556', 'n02002724', 'n02017213', 'n02033041', 
         'n02085620', 'n02085782', 'n02086079', 'n02087046', 'n02087394', 
         'n02088364', 'n02093859', 'n02097130', 'n02097474', 'n02098105', 
         'n02098286', 'n02098413', 'n02102040', 'n02102177', 'n02105641', 
         'n02106166', 'n02106382', 'n02106550', 'n02107142', 'n02110185', 
         'n02110806', 'n02111277', 'n02111500', 'n02111889', 'n02113624', 
         'n02115913', 'n02116738', 'n02125311', 'n02128925', 'n02129165', 
         'n02134418', 'n02168699', 'n02174001', 'n02177972', 'n02259212', 
         'n02268443', 'n02281787', 'n02328150', 'n02342885', 'n02361337', 
         'n02395406', 'n02396427', 'n02410509', 'n02415577', 'n02417914', 
         'n02422699', 'n02486261', 'n02488291', 'n02488702', 'n02514041', 
         'n02687172', 'n02708093', 'n02782093', 'n02791124', 'n02815834', 
         'n02817516', 'n02837789', 'n02870880', 'n02906734', 'n02910353', 
         'n02927161', 'n02948072', 'n03014705', 'n03026506', 'n03028079', 
         'n03065424', 'n03075370', 'n03095699', 'n03110669', 'n03124043', 
         'n03160309', 'n03188531', 'n03201208', 'n03250847', 'n03255030', 
         'n03272010', 'n03344393', 'n03347037', 'n03388043', 'n03393912', 
         'n03417042', 'n03424325', 'n03444034', 'n03445777', 'n03461385', 
         'n03485407', 'n03498962', 'n03529860', 'n03535780', 'n03584829', 
         'n03595614', 'n03599486', 'n03637318', 'n03662601', 'n03666591', 
         'n03697007', 'n03706229', 'n03709823', 'n03721384', 'n03743016', 
         'n03769881', 'n03775071', 'n03782006', 'n03785016', 'n03832673', 
         'n03841143', 'n03873416', 'n03891332', 'n03895866', 'n03903868', 
         'n03908618', 'n03944341', 'n03950228', 'n03954731', 'n03961711', 
         'n03970156', 'n03976467', 'n03982430', 'n04009552', 'n04049303', 
         'n04067472', 'n04152593', 'n04154565', 'n04162706', 'n04264628', 
         'n04265275', 'n04266014', 'n04311004', 'n04311174', 'n04328186', 
         'n04330267', 'n04344873', 'n04347754', 'n04357314', 'n04367480', 
         'n04371774', 'n04399382', 'n04417672', 'n04428191', 'n04435653', 
         'n04443257', 'n04479046', 'n04483307', 'n04486054', 'n04501370', 
         'n04507155', 'n04548362', 'n04552348', 'n04554684', 'n04557648', 
         'n04562935', 'n04579145', 'n04590129', 'n04604644', 'n04606251', 
         'n06785654', 'n06874185', 'n07565083', 'n07583066', 'n07615774', 
         'n07695742', 'n07715103', 'n07716906', 'n07718472', 'n07734744', 
         'n07742313', 'n07745940', 'n07860988', 'n07932039', 'n09288635', 
         'n09399592', 'n09472597', 'n13037406', 'n13044778', 'n13054560']
IN200_words = ['rooster', 'brambling', 'vulture', 'fire salamander', 'agama', 
            'Komodo dragon', 'American alligator', 'smooth green snake', 'yellow garden spider', 'barn spider', 
            'centipede', 'peafowl', 'coucal', 'hornbill', 'black swan', 
            'echidna', 'wallaby', 'wombat', 'flatworm', 'conch', 
            'fiddler crab', 'white stork', 'black stork', 'common gallinule', 'dowitcher', 
            'Chihuahua', 'Japanese Chin', 'Pekingese', 'toy terrier', 'Rhodesian Ridgeback', 
            'Beagle', 'Kerry Blue Terrier', 'Giant Schnauzer', 'Tibetan Terrier', 'Soft-coated Wheaten Terrier', 
            'West Highland White Terrier', 'Lhasa Apso', 'English Springer Spaniel', 'Welsh Springer Spaniel', 'Old English Sheepdog', 
            'Border Collie', 'Bouvier des Flandres dog', 'Rottweiler', 'Dobermann', 'Siberian Husky', 
            'Basenji', 'Newfoundland dog', 'Great Pyrenees dog', 'Samoyed', 'Toy Poodle', 
            'dhole', 'African wild dog', 'cougar', 'jaguar', 'lion', 
            'sloth bear', 'longhorn beetle', 'rhinoceros beetle', 'weevil', 'leafhopper', 
            'dragonfly', 'gossamer-winged butterfly', 'Angora rabbit', 'hamster', 'marmot', 
            'pig', 'wild boar', 'bison', 'bighorn sheep', 'Alpine ibex', 
            'impala (antelope)', 'patas monkey', 'langur', 'black-and-white colobus', 'snoek fish', 
            'aircraft carrier', 'analog clock', 'balloon', 'barber chair', 'beaker', 
            'military hat (bearskin or shako)', 'bikini', 'bookcase', 'broom', 'buckle', 
            'butcher shop', 'candle', 'storage chest', 'Christmas stocking', 'church', 
            'spiral or coil', 'combination lock', 'container ship', 'cornet', 'cowboy boot', 
            'dam', 'diaper', 'dining table', 'drumstick', 'dumbbell', 
            'electric guitar', 'fireboat', 'fire screen', 'fountain', 'freight car', 
            'garbage truck', 'gas mask or respirator', 'go-kart', 'golf ball', 'grocery store', 
            'hand-held computer', 'hatchet', 'home theater', 'gymnastic horizontal bar', 'clothes iron', 
            'T-shirt', 'rickshaw', 'lampshade', 'lifeboat', 'lighter', 
            'sawmill', 'magnetic compass', 'messenger bag', 'marimba', 'megalith', 
            'minibus', 'mitten', 'monitor', 'moped', 'notebook computer', 
            'odometer', 'paddle', 'parking meter', 'railroad car', 'pedestal', 
            'pencil case', 'pinwheel', 'drink pitcher', 'block plane', 'plate rack', 
            'plunger', 'Polaroid camera', 'pool table', 'projector', 'rain barrel', 
            'fishing casting reel', 'CRT monitor', 'screwdriver', 'seat belt', 'keyboard space bar', 
            'space heater', 'space shuttle', 'through arch bridge', 'steel drum', 'stopwatch', 
            'stove', 'couch', 'submarine', 'sunscreen', 'mop', 
            'swing', 'teddy bear', 'thatched roof', 'threshing machine', 'tile roof', 
            'tobacco shop', 'trench coat', 'trimaran', 'triumphal arch', 'turnstile', 
            'umbrella', 'wallet', 'military aircraft', 'washing machine', 'water bottle', 
            'water tower', 'whiskey jug', 'window shade', 'split-rail fence', 'shipwreck', 
            'crossword', 'traffic light', 'menu', 'guacamole', 'popsicle', 
            'pretzel', 'cauliflower', 'spaghetti squash', 'cucumber', 'mushroom', 
            'Granny Smith apple', 'strawberry', 'dough', 'eggnog', 'geyser', 
            'promontory', 'volcano', 'gyromitra', 'earth star fungus', 'bolete']

IN200_wnids, IN200_words = zip(*sorted(zip(IN200_wnids, IN200_words)))
IN200_wnids, IN200_words = list(IN200_wnids), list(IN200_words)

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
def semantic_filter_rank(embeddings, labels, args, words=None):
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

        top_num = int(length_sim * args.filter_ratio)
        best_half = sims[:top_num]
        worst_half = sims[top_num:]
        
        for _, idx in best_half:
            idxs.append(idx)  # half best
        for _, idx in worst_half:
            remove_idxs.append(idx)  # half worst
            print("prompt: {}".format(f"a photo of a {words[labels[idx]]}"))
            print("removed img: ", clip_dataset.samples[idx][0])
                
    # predictions = torch.tensor(predictions).to("cuda")
    print(f"kept length: {len(idxs)}")
    print(f"removed length: {len(remove_idxs)}")
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
elif hasattr(args, "data_pt"):
    aug_data = torch.load(args.data_pt)
    aug_emb, aug_labels = aug_data["clip_embeddings"], aug_data["labels"]
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
elif args.data.base_dataset in ["IN200_sub", "PETs"]:
   words = IN200_words
print(f"words: {words}")

# aug_labels, base_labels = [int(s[1]) for s in dataset.samples], [int(s[1]) for s in trainset.samples]

##########################################################################################################
################################ Find semantic errors ####################################################
##########################################################################################################
# predictions, semantic_filtered, kept_idxs = semantic_filter_SEI(aug_emb, aug_labels, list(args.filter.negative_prompts))
semantic_filtered, kept_idxs = semantic_filter_rank(aug_emb, aug_labels, args, words=words)


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