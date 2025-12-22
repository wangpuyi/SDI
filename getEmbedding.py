import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image

from ip_adapter import SEIImageProjModel
from torchvision import transforms
import time, json, os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--total_split", type=int, default=4)
parser.add_argument("--split", type=int, default=0)
parser.add_argument("--projector_ckpt", type=str)
parser.add_argument("--data_root", type=str)
parser.add_argument("--save_root", type=str)
args = parser.parse_args()
print(args)

def change_postfix(save_root, category, postfix):
    parts = os.path.join(save_root, category).split(".")
    parts[-1] = postfix
    return ".".join(parts)
    
# projector_ckpt =  # IN100
projector_ckpt = args.projector_ckpt

image_encoder_path = "/grp01/cs_hszhao/cs002u03/ckpt/clip-vit-large-patch14"

device = "cuda"

image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(device)
clip_image_processor = CLIPImageProcessor()

image_proj_model = SEIImageProjModel(
    cross_attention_dim=768,
    clip_embeddings_dim=image_encoder.config.projection_dim,
).to(device) 
image_proj_model.load_state_dict(torch.load(projector_ckpt, map_location=device))

def get_image_embeds(raw_image):
    # img_path = '/mnt/petrelfs/share/imagenet/images/train/n01498041/n01498041_28.JPEG'
    # raw_image = Image.open(img_path).convert("RGB")
    clip_image = clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
    image_embeds = image_encoder(clip_image.to(device)).image_embeds
    projected_image_embed = image_proj_model(image_embeds)

    return projected_image_embed

# print(projected_image_embed)
# print(projected_image_embed.shape) # torch.Size([1, 768])
# print(projected_image_embed.device) # cuda:0

data_root = args.data_root
wnids = os.listdir(data_root)
img_path_list = []
for wnid in wnids:
    img_path_list += [os.path.join(wnid, item) for item in os.listdir(os.path.join(data_root, wnid))]
print(f"total images: {len(img_path_list)}")

if args.total_split > 1 and args.split != args.total_split-1:
    split_size = len(img_path_list) // args.total_split
    img_path_list = img_path_list[args.split*split_size:(args.split+1)*split_size]
    print(f"begin: {args.split*split_size} end: {(args.split+1)*split_size}")
elif args.total_split > 1 and args.split == args.total_split-1:
    split_size = len(img_path_list) // args.total_split
    img_path_list = img_path_list[args.split*split_size:]
    print(f"begin: {args.split*split_size} end: End")

print(f"split images: {len(img_path_list)}")

save_root = args.save_root

# extentions = [".jpg",".jpeg",".png"]
start = time.perf_counter()
for name in tqdm(img_path_list):
    class_name = name.split("/")[0] # "n09332890/n09332890_18273.JPEG"
    img_path = os.path.join(data_root, name)
    save_pt_path = change_postfix(save_root, name, "pt")
    raw_image = Image.open(img_path).convert("RGB")
    img_embed = get_image_embeds(raw_image).squeeze().detach().to("cpu") # torch.Size([768])
    if not os.path.exists(os.path.dirname(save_pt_path)):
        os.makedirs(os.path.dirname(save_pt_path), exist_ok=True)
    torch.save(img_embed, save_pt_path)

time_cost = time.perf_counter() - start
print(f"Finish in {time_cost} seconds")