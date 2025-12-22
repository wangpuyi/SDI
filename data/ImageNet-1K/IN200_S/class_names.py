import os

wnids = os.listdir("/grp01/cs_hszhao/cs002u03/dataset/IN200_S/train")
wnids = sorted(wnids)
Wnids2Categories = {}
with open('/home/cs002u03/SEI/data/ImageNet-1K/classnames.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        parts = line.split(" ")
        wnid, category = parts[0], " ".join(parts[1:])
        if wnid in wnids:
            Wnids2Categories[wnid]= category

print(Wnids2Categories)
print(len(Wnids2Categories))

wnids = []
categories = []
for wnid, category in Wnids2Categories.items():
    wnids.append(wnid)
    categories.append(category)
for i in range(0, len(categories), 5):
    print(f"'{categories[i]}', '{categories[i+1]}', '{categories[i+2]}', '{categories[i+3]}', '{categories[i+4]}', ")
print(len(wnids))
print(len(categories))
