import os
import json, random
import scipy.io

def getCategories():
    '''
    return wnid to name e.g.
        categories = [
        ("n01443537", "goldfish"),
        ("n01614925", "bald eagle"),
        ]
    '''
    meta = scipy.io.loadmat('/mnt/hwfile/gveval/yangshuo/dataset/ImageNet-1k/ILSVRC2012_devkit_t12/data/meta.mat')
    synsets = meta['synsets']
    img_root = "/mnt/petrelfs/share/imagenet/images/train"
    wnid_list = os.listdir(img_root)

    # for entry in synsets[:1]:
    #     print(entry)
    #     print(entry['words'])

    categories = []

    for entry in synsets:
        words = entry['words'][0][0]
        id = entry['ILSVRC2012_ID'][0][0]
        wnid = entry['WNID'][0][0]
        if wnid in wnid_list:
            categories.append((wnid, words.split(",")[0]))
        
    return categories

# 假设ImageNet-1k验证集数据的根目录
val_dir = "/mnt/petrelfs/share/imagenet/images/val"
meta_file = "/mnt/petrelfs/share/imagenet/images/meta/val.txt"

# 挑选的类别及其对应的标签ID
# categories = ["n01443537", "n01614925", "n11939491", "n07747607", "n04153751", "n03476684", "n04507155", "n04285008", "n02690373", "n09332890"]
# category_ids = [1, 22, 985, 950, 783, 584, 879, 817, 404, 975]  # 将类别映射为0-9的标签
categories = getCategories()
categories_3 = []
for idx, item in enumerate(sorted(categories, key=lambda x: x[0])):
    categories_3.append((item[0], item[1], idx))

random.seed(42)
categories = random.sample(categories, 100)

label1kTwowind = {int(item[2]): item[0] for item in categories_3} # label-1k: wnid | item: ('n11939491', 'daisy')

# 读取原始验证集元数据文件
val_data = []
with open(meta_file, "r") as f:
    lines = f.readlines()
    for line in lines:
        image_file, label = line.strip().split()
        label = int(label)
        val_data.append((image_file, label))

# 生成标签映射
label_map = {item[0]: idx for idx, item in enumerate(sorted(categories, key=lambda x: x[0]))} # wnid: idx | from 0 to (len-1) 
wnids_sub = [item[0] for item in categories]
category_ids = [item[2] for item in categories_3 if item[0] in wnids_sub]

# 筛选指定类别的图像
selected_data = []
for image_file, label in val_data:
    if label in category_ids:
        wnid = label1kTwowind[label]
        selected_data.append({
            "image_file": os.path.join("/mnt/petrelfs/share/imagenet/images/val", image_file),
            "label": label_map[wnid],
            "label-1k": label
        })

print(len(selected_data))

# 保存为JSON文件
with open("data/ImageNet-1K/val_100.json", "w") as f:
    json.dump(selected_data, f)