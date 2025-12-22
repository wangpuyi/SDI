import os, io
from tqdm import tqdm
import torch as torch
import torchvision
from torchvision import transforms
import numpy as np
import torchvision.datasets as dsets
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torchvision.transforms.functional import crop
from collections import Counter

import datasets
from datasets.base import *
from datasets import *
# from cutmix.cutmix import CutMix

def crop_wilds(image):
    return crop(image, 10, 0, 400, 448)

class Cutout(object):
    """Randomly mask out one or more patches from an image."""
    def __init__(self, n_holes, length):
        self.n_holes = n_holes  # number of holes to cut out from the image
        self.length = length  # length of the holes

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Input image to apply Cutout.

        Returns:
            PIL.Image: Image with Cutout applied.
        """
        if not isinstance(img, Image.Image):  # 检查是否是 PIL 图像
            raise TypeError(f"Expected PIL.Image but got {type(img)}")
        
        # 将 PIL Image 转换为 numpy 数组
        img = np.array(img)

        h, w = img.shape[0], img.shape[1]  # 获取图像高度和宽度
        mask = np.ones((h, w), np.float32)  # 创建全 1 的 mask

        for _ in range(self.n_holes):
            # 随机选择要裁剪的方形区域的中心点
            y = np.random.randint(h)
            x = np.random.randint(w)

            # 计算裁剪区域的边界
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            # 将 mask 的裁剪区域置为 0
            mask[y1:y2, x1:x2] = 0.

        # 扩展 mask 维度，适应图像的通道维度 (H, W, C)
        mask = np.expand_dims(mask, axis=-1)

        # 将图像乘以 mask，应用遮挡
        img = img * mask

        # 将 numpy 数组转换回 PIL Image
        img = Image.fromarray(np.uint8(img))

        return img

class GridMask(object):
    """
    Apply GridMask data augmentation, which randomly masks out square regions
    in an image to create a grid-like pattern.

    Args:
        use_h (bool): Whether to apply masking along the height.
        use_w (bool): Whether to apply masking along the width.
        rotate (int): Rotation angle in degrees for the grid lines.
        offset (bool): Whether to apply random offsets to the grid.
        ratio (float): Ratio of masked squares in the grid.
        mode (int): Masking mode, 0 or 1.
        prob (float): Probability of applying this augmentation.
    """

    def __init__(self, use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5, mode=1, prob=1.0):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img

        # 检查输入类型是否为 PIL.Image
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected input to be a PIL.Image, but got {type(img)}")

        # 将 PIL.Image 转换为 numpy 数组
        img = np.array(img)

        h, w = img.shape[0], img.shape[1]  # 获取图像的高度和宽度
        hh = int(1.5 * h)
        ww = int(1.5 * w)

        # 随机选择网格大小 d，并计算遮蔽区域的大小 l
        d = np.random.randint(90, 140)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)

        # 创建全 1 的 mask
        mask = np.ones((hh, ww), np.float32)

        # 随机选择起始位置
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        # 使用 h 和 w 方向的遮蔽
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] = 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] = 0

        # 随机旋转 mask
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask * 255))  # 转换为 PIL 图像
        mask = mask.rotate(r)
        mask = np.asarray(mask) / 255.0  # 转回 numpy 数组

        # 裁剪出与输入图像相同的大小
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        # 处理不同的模式
        if self.mode == 1:
            mask = 1 - mask

        # 将 mask 应用到图像
        mask = np.expand_dims(mask, axis=-1)
        img = img * mask

        # 将 numpy 数组转换回 PIL.Image 格式
        img = Image.fromarray(np.uint8(img))

        return img

def get_train_transform(dataset_name="Imagenet", model=None, augmentation=None):
    """"
    Gets the transform for a given dataset
    """
    # any data augmentation happens here
    transform_list = []
    if augmentation == "augmix":
        print("Applying AugMix")
        transform_list.append(transforms.AugMix())
    if augmentation == "color-jitter":
        print("Applying color jitter")
        transform_list.append(transforms.ColorJitter(brightness=.5, hue=.3))
    if augmentation == "randaug":
        print("Applying RandAug augmentations")
        transform_list.append(transforms.RandAugment())
    if augmentation == "auto":
        print("Applying automatic augmentations")
        transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
    if augmentation == 'rotation':
        print("Applying scale jitter")
        transform_list.append(transforms.RandomRotation(10))
    if augmentation == 'scale_jitter':
        print("Applying scale jitter")
        transform_list.append(transforms.v2.ScaleJitter(target_size=(224, 224)))
    if augmentation == 'color_jitter':
        print("Applying color jitter")
        transform_list.append(transforms.ColorJitter(brightness=.5, hue=.3))
    if augmentation == 'cutout':
        print("Applying Cutout")
        # 使用自定义 Cutout 类，裁剪随机遮挡区域
        transform_list.append(Cutout(n_holes=1, length=32))
    if augmentation == 'gridmask':
        print("Applying GridMask")
        transform_list.append(GridMask(use_h=True, use_w=True, rotate=15, ratio=0.5, prob=1.0))

    # standard preprocessing
    if model in ['RN50', 'ViT-B/32']: # if we are evaluating a clip model we use its transforms
        print("...loading CLIP model")
        net, transform = clip.load(model)
    elif "iWildCam" in dataset_name:
        transform_list += [transforms.ToTensor(),
                                #   transforms.Grayscale(num_output_channels=3),
                                  transforms.Resize((448, 448)),
                                  transforms.Lambda(crop_wilds),
                                  transforms.Resize((224, 224))]
    else:
        # transform_list += [
        #     # transforms.Resize((224,224)),
        #     transforms.RandomResizedCrop(224),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ]
        transform_list += [
        transforms.Resize((256,256)),
        transforms.RandomRotation(15,),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]

    
    return transforms.Compose(transform_list)

def get_val_transform(dataset_name="Imagenet", model=None):
    """"
    Gets the transform for a given dataset
    """
    transform_list = []
    if model in ['RN50', 'ViT-B/32']: # if we are evaluating a clip model we use its transforms
        print("...loading CLIP model")
        net, transform = clip.load(model)
    elif "iWildCam" in dataset_name:
        transform_list += [transforms.ToTensor(),
                                #   transforms.Grayscale(num_output_channels=3),
                                  transforms.Resize((448, 448)),
                                  transforms.Lambda(crop_wilds),
                                  transforms.Resize((224, 224))]
    else:
        # transform_list += [
        #     transforms.Resize((224,224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ]
        transform_list += [
        transforms.Resize((256,256)),
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]

    
    return transforms.Compose(transform_list)

def get_dataset(dataset_name, transform, val_transform, root='./data', embedding_root=None):
    if dataset_name == "Waterbirds" or dataset_name == 'WaterbirdsExtra': # change these data paths
        trainset = Waterbirds(root=root, split='train', transform=transform)
        train_ids = trainset.get_subset(groups=[0,3], num_per_class=1000)
        # get every 4th idx
        train_ids = train_ids[::4]
        train_extra_ids = trainset.get_subset(groups=[1,2], num_per_class=1000)
        extra_trainset = Subset(trainset, train_extra_ids)
        trainset = Subset(trainset, train_ids) #100% biased
        valset = Waterbirds(root=root, split='val', transform=val_transform)
        idxs = valset.get_subset(groups=[0,3], num_per_class=1000)
        extra_idxs = valset.get_subset(groups=[1,2], num_per_class=1000)
        extra_valset = Subset(valset, extra_idxs)
        extraset = CombinedDataset([extra_valset, extra_trainset])
        valset = Subset(valset, idxs)
        testset = Waterbirds(root=root, split='test', transform=val_transform)
        if dataset_name == 'WaterbirdsExtra':
            trainset = CombinedDataset([trainset, extraset])
    elif dataset_name == "iWildCamMini" or dataset_name == "iWildCamMiniExtra":
        trainset = WILDS(root=f'{root}/iwildcam_v2.0/train', split='train', transform=transform)
        valset = WILDS(root=f'{root}/iwildcam_v2.0/train', split='val', transform=val_transform)
        testset = WILDS(root=f'{root}/iwildcam_v2.0/train', split='test', transform=val_transform)
        extraset = WILDS(root=f'{root}/iwildcam_v2.0/train', split='train_extra', transform=transform)
        if dataset_name == 'iWildCamMiniExtra':
            trainset = CombinedDataset([trainset, extraset])
    elif dataset_name == 'Cub2011':
        trainset = Cub2011(root=root, split='train', transform=transform)
        valset = Cub2011(root=root, split='val', transform=val_transform)
        extraset = trainset
        testset = valset
        # if dataset_name == 'Cub2011Extra':
        #     trainset = CombinedDataset([trainset, extraset])
    elif dataset_name == 'Cifar10_sub':
        trainset = Cifar10(root=root, split='train', subset=True, transform=transform)
        valset = Cifar10(root=root, split='val', transform=val_transform)
        testset = valset
        extraset = Cifar10(root=root, subset=True, split='extra', transform=transform)
    elif dataset_name == 'Caltech100' or dataset_name == 'Caltech100Extra':
        trainset = Caltech100(root=root, split='train', transform=transform)
        valset = Caltech100(root=root, split='val', transform=val_transform)
        extraset = Caltech100(root=root, split='extra', transform=transform)
        testset = valset
        if dataset_name == 'Caltech100Extra':
            trainset = CombinedDataset([trainset, extraset])
    elif dataset_name == 'IN10':
        trainset = IN10(root=root, split='train', transform=transform)
        valset = IN10(root=root, split='val', transform=val_transform)
        extraset = None
        testset = valset
    elif dataset_name == 'IN100':
        trainset = IN100(root=root, split='train', transform=transform)
        valset = IN100(root=root, split='val', transform=val_transform)
        extraset = trainset
        testset = valset
    elif dataset_name == 'IN100_sub':
        trainset = IN100_sub(root=root, split='train', transform=transform)
        valset = IN100_sub(root=root, split='val', transform=val_transform)
        extraset = trainset
        testset = valset
    elif dataset_name == 'IN200_sub':
        trainset = IN100_sub(root=root, split='train', transform=transform)
        valset = IN100_sub(root=root, split='val', transform=val_transform)
        extraset = trainset
        testset = valset
    elif dataset_name == "Cars196":
        trainset = Cars(root=root, split='train', transform=transform)
        valset = Cars(root=root, split='val', transform=val_transform)
        extraset = trainset
        testset = valset
    elif dataset_name == "PETs":
        trainset = PETs(root=root, split='train', transform=transform)
        valset = PETs(root=root, split='val', transform=val_transform)
        extraset = trainset
        testset = valset
    elif dataset_name == "Flowers102":
        trainset = Flowers(root=root, split='train', transform=transform)
        valset = Flowers(root=root, split='val', transform=val_transform)
        extraset = trainset
        testset = valset
    elif dataset_name == "DTD":
        trainset = DTD(root=root, split='train', transform=transform)
        valset = DTD(root=root, split='val', transform=val_transform)
        extraset = trainset
        testset = valset
    elif dataset_name == "Caltech101":
        trainset = Caltech101(root=root, split='train', transform=transform)
        valset = Caltech101(root=root, split='val', transform=val_transform)
        extraset = trainset
        testset = valset        
    elif dataset_name == "BreastMNIST":
        trainset = BreastMNIST(root=root, split='train', transform=transform)
        valset = BreastMNIST(root=root, split='val', transform=val_transform)
        extraset = trainset
        testset = valset
    elif dataset_name == "OrgansMNIST":
        trainset = OrgansMNIST(root=root, split='train', transform=transform)
        valset = OrgansMNIST(root=root, split='val', transform=val_transform)
        extraset = trainset
        testset = valset
    elif dataset_name == "IN200_sub_generalize":
        trainset = IN200_sub_generalize(root=root, transform=transform)
        valset = trainset
        extraset = trainset
        testset = trainset

        
    if embedding_root:
        trainset = EmbeddingDataset(os.path.join(embedding_root, dataset_name), trainset, split='train')
        valset = EmbeddingDataset(os.path.join(embedding_root, dataset_name), valset, split='val')
        testset = EmbeddingDataset(os.path.join(embedding_root, dataset_name), testset, split='test')

    # assert that the trainset has the attributes groups, labels, and class_names
    for var in ['groups', 'targets', 'group_names', 'class_names', 'class_weights']:
        assert all([hasattr(dataset, var) for dataset in [trainset, valset, testset]]), f"datasets missing the attribute {var}"

    return trainset, valset, testset, extraset

def get_filtered_dataset(args, transform, val_transform):
    np.random.seed(args.seed)
    trainset, valset, testset, extraset = get_dataset(args.data.base_dataset, transform, val_transform, root=args.data.base_root, embedding_root=args.data.embedding_root if args.model == 'MLP' else None)
    if args.data.extra_dataset and not args.eval_only: # defalut (false, false)
        dataset = get_edited_dataset(args, transform)
        if args.data.num_extra == 'extra':
            dataset = subsample(extraset, dataset, multiples=args.data.multiples) # make sure we are sampling the same number of images as the extraset
        elif type(args.data.num_extra) == int: # randomly sample x images from the dataset
            print("sampled", args.data.num_extra, "images from the extra dataset")
            if args.data.class_balance:
                dataset = get_class_balanced_subset(dataset, args.data.num_extra // len(dataset.classes))
            else:
                dataset = Subset(dataset, np.random.choice(len(dataset), args.data.num_extra, replace=True))
        print(f"Added extra data with class counts {Counter(dataset.targets)}")
        if args.data.extra_only:
            trainset = dataset
        else:
            trainset = CombinedDataset([trainset, dataset])

#        if args.data.augmentation == 'cutmix': # hacky way to add cutmix augmentation
#            trainset = CutMix(trainset, num_class=len(trainset.classes), beta=1.0, prob=0.5, num_mix=2).dataset
    return trainset, valset, testset

def get_edited_dataset(args, transform, full=False):
    if type(args.data.extra_root) != str:
        print("Roots", list(args.data.extra_root))
        roots, dsets = list(args.data.extra_root), []
        for i, r in enumerate(roots):
            dsets.append(getattr(datasets.base, args.data.extra_dataset)(r, transform=transform, cfg=args, group=i))
        dataset = CombinedDataset(dsets)
    else: # extra_dataset is the dataset type for the augmented data (usually it's Img2ImgDataset or BasicDataset)
        dataset = getattr(datasets.base, args.data.extra_dataset)(args.data.extra_root, transform=transform, cfg=args)

    if args.data.filter: # default True
        path = f'{args.filter.save_dir}/{args.name}/filtered_idxs/kept.npy' if not args.filter.filtered_path else args.filter.filtered_path
        if os.path.exists(path):
            filtered_idxs = np.load(path)
        else:
            raise ValueError(f"can't find file {path}")
        print(f"Filtering kept {len(filtered_idxs)} out of {len(dataset)} images")
        dataset = Subset(dataset, filtered_idxs)

    if full or args.data.extra_dataset != 'Img2ImgDataset':
        return dataset
    
    sample_groups = {}
    for i, s in enumerate(dataset.samples):
        filename = s[0].split("/")[-1] # maybe generated images has different format \ get {idx}-{j}.jpg 
        if len(filename.split('.')[0].split("-")) == 1:
            print(f"skipping {filename}")
            continue
        else:
            idx, j = filename.split('.')[0].split("-")[0], filename.split('.')[0].split("-")[1]
        if idx not in sample_groups:
            sample_groups[idx] = [(s[0], s[1], i)]
        else:
            sample_groups[idx].append([s[0], s[1], i])

    chosen_idxs = []
    for k, v in sample_groups.items():
        # randomly select **one** sample from each group
        chosen = np.random.choice(list(range(len(v))), replace=False)
        chosen_idxs.append(v[chosen][2])

    dataset = Subset(dataset, chosen_idxs)
    return dataset
