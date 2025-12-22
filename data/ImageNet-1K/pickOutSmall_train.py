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

root_dir = "/mnt/petrelfs/share/imagenet/images/train"

# 挑选的类别
# categories = ["n01443537", "n01614925", "n11939491", "n07747607", "n04153751", "n03476684", "n04507155", "n04285008", "n02690373",
#           "n09332890"]
# [("n01443537", "goldfish"),("n01614925", "bald eagle")]
categories = getCategories() #
wnid2word_map = {}
for item in categories:
    wnid2word_map[item[0]] = item[1]
random.seed(42)
categories = random.sample(categories, 100)

data = []
label_map = {item[0]: idx for idx, item in enumerate(sorted(categories))} # wnid to label(0-99)

for item in sorted(categories):
    category = item[0]
    category_dir = os.path.join(root_dir, category)
    images = os.listdir(category_dir)[:200]  # 取前200张图片
    for image in images:
        data.append({
            "image_file": os.path.join(category, image),
            "label": label_map[category],
            "word": wnid2word_map[category]
        })

# 保存JSON文件
with open("data/ImageNet-1K/train_100.json", "w") as f:
    json.dump(data, f)