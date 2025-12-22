import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image

from ip_adapter import SEIImageProjModel
from torchvision import transforms
import numpy as np


base_model_path = "/grp01/cs_hszhao/cs002u03/ckpt/stable-diffusion-v1-5"
vae_model_path = "/grp01/cs_hszhao/cs002u03/ckpt/sd-vae-ft-mse"

#PETs
projector_ckpt = "/grp01/cs_hszhao/cs002u03/ckpt/adapter/PETs/bs64-lr_0.0005-20240927-222310/checkpoint-99.pt"
#CUB
projector_ckpt = "/grp01/cs_hszhao/cs002u03/ckpt/adapter/CUB/bs64-lr_0.0005-20241004-205514/checkpoint-99.pt"
# IN100
projector_ckpt = "/grp01/cs_hszhao/cs002u03/ckpt/adapter/IN100Sub/bs64-lr_0.0005-20240927-174903/checkpoint-99.pt"
# flower
projector_ckpt = "/grp01/cs_hszhao/cs002u03/ckpt/adapter/Flowers102/bs64-lr_0.0005-20240930-212450/checkpoint-99.pt" 

# PETs
img_path = "/grp01/cs_hszhao/cs002u03/dataset/PETs/trainval/British_Shorthair/British_Shorthair_109.jpg"
img_path = "/grp01/cs_hszhao/cs002u03/dataset/PETs/trainval/British_Shorthair/British_Shorthair_128.jpg"
img_path = "/grp01/cs_hszhao/cs002u03/dataset/PETs/trainval/British_Shorthair/British_Shorthair_17.jpg"

# CUB
img_path = "/grp01/cs_hszhao/cs002u03/dataset/CUB_200_2011/train/001.Black_footed_Albatross/Black_Footed_Albatross_0017_796098.jpg"
img_path = "/grp01/cs_hszhao/cs002u03/dataset/CUB_200_2011/train/001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg"
img_path = "/grp01/cs_hszhao/cs002u03/dataset/CUB_200_2011/train/001.Black_footed_Albatross/Black_Footed_Albatross_0038_212.jpg"

# IN200
img_path = "/grp01/cs_hszhao/cs002u03/dataset/IN200_S/train/n03388043/n03388043_5873.JPEG"

# IN100
img_path = "/grp01/cs_hszhao/cs002u03/dataset/IN100_sub/train/n03594945/n03594945_183.JPEG"
img_path = "/grp01/cs_hszhao/cs002u03/dataset/IN100_sub/train/n03594945/n03594945_681.JPEG"
img_path = "/grp01/cs_hszhao/cs002u03/dataset/IN100_sub/train/n03594945/n03594945_290.JPEG"
img_path = "/grp01/cs_hszhao/cs002u03/dataset/IN100_sub/train/n03594945/n03594945_354.JPEG"
img_path = "/grp01/cs_hszhao/cs002u03/dataset/IN100_sub/train/n03594945/n03594945_719.JPEG"
# img_path = "/grp01/cs_hszhao/cs002u03/dataset/IN100_sub/train/n03594945/n03594945_2785.JPEG"

# flowers
img_path = "/grp01/cs_hszhao/cs002u03/dataset/Flower102/train/54_sunflower/image_05401.jpg"
img_path = "/grp01/cs_hszhao/cs002u03/dataset/Flower102/train/01_pink_primrose/image_06742.jpg"
img_path = "/grp01/cs_hszhao/cs002u03/dataset/Flower102/train/01_pink_primrose/image_06751.jpg"
img_path = "/grp01/cs_hszhao/cs002u03/dataset/Flower102/train/01_pink_primrose/image_06747.jpg"

device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(device, dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
).to(device)
image_encoder_path = "/grp01/cs_hszhao/cs002u03/ckpt/clip-vit-large-patch14"
image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(device)
clip_image_processor = CLIPImageProcessor()
text_tokenizer = pipe.tokenizer
text_model = pipe.text_encoder



raw_image = Image.open(img_path).convert("RGB")
clip_image = clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
image_embeds = image_encoder(clip_image.to(device)).image_embeds
image_proj_model = SEIImageProjModel(
    cross_attention_dim=768,
    clip_embeddings_dim=image_encoder.config.projection_dim,
).to(device) 
image_proj_model.load_state_dict(torch.load(projector_ckpt))

projected_image_embed = image_proj_model(image_embeds)
#size
print(projected_image_embed.shape) # 

prompt = "a high quality photo of a class." # class is a placeholder
# prompt = "class"
neg_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
text_inputs = text_tokenizer(prompt, return_tensors="pt", max_length=77, padding="max_length", truncation=True)["input_ids"]
tokens = text_tokenizer.convert_ids_to_tokens(text_inputs[0])
word_index = tokens.index("class</w>")
print(f"tokens: {tokens}")
print(f"index: {word_index}")
text_outputs = text_model(input_ids=text_inputs.to(device))
text_embeds = text_outputs.last_hidden_state
print(f"text_embeds shape: {text_embeds.shape}")
text_embeds[:, word_index, :] = projected_image_embed

neg_text_inputs = text_tokenizer(neg_prompt, return_tensors="pt", max_length=77, padding="max_length", truncation=True)["input_ids"]
neg_text_outputs = text_model(input_ids=neg_text_inputs.to(device))
neg_text_embeds = neg_text_outputs.last_hidden_state

# Generate the image
num = 5
# images = pipe(prompt_embeds=text_embeds.repeat(num,1,1)).images
images = pipe(prompt_embeds=text_embeds.repeat(num,1,1), negative_prompt_embeds=neg_text_embeds.repeat(num,1,1)).images

image_arrays = [np.array(img) for img in images]
height, width, _ = image_arrays[0].shape

combined_image = Image.new('RGB', (width * len(images), height))

for i, img in enumerate(images):
    combined_image.paste(img, (i * width, 0))

combined_image.save('combined_image.png')
# images[0].save(infer_test.png)