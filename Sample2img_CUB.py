import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image

from ip_adapter import SEIImageProjModel
from torchvision import transforms
import os
import time
import json


base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"

projector_ckpt = "/mnt/hwfile/gveval/yangshuo/ckpt/adapter/CUB/bs32-lr_0.0005-20240830-114019/checkpoint-90.pt"

image_encoder_path = "openai/clip-vit-large-patch14"

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
image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_image_processor = CLIPImageProcessor()
text_tokenizer = pipe.tokenizer
text_model = pipe.text_encoder

image_proj_model = SEIImageProjModel(
    cross_attention_dim=768,
    clip_embeddings_dim=image_encoder.config.projection_dim,
).to(device) 
image_proj_model.load_state_dict(torch.load(projector_ckpt))

def get_image_embeds(raw_image):
    clip_image = clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
    image_embeds = image_encoder(clip_image.to(device)).image_embeds
    projected_image_embed = image_proj_model(image_embeds)

    return projected_image_embed

data_root = "/mnt/hwfile/gveval/yangshuo/dataset/Caltech-UCSD-Birds-200-2011"

class_name = "Black_footed_Albatross" # tripod

json_list = json.load(open("data/CUB/train.json"))
img_path_list = [os.path.join(data_root, item["image_file"]) for item in json_list if class_name in item["image_file"]]

extentions = [".jpg",".jpeg",".png"]
tensors = []
start = time.perf_counter()
with torch.no_grad():
    for name in img_path_list:
        # save_img_path = os.path.join(save_root, class_name, name)
        img_path = os.path.join(data_root, name)
        raw_image = Image.open(img_path).convert("RGB")
        projected_image_embed = get_image_embeds(raw_image)
        tensors.append(projected_image_embed.cpu().numpy().squeeze())
    time_cost = time.perf_counter() - start
    print(f"Finish in {time_cost} seconds")

from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal  
import numpy as np
arrays = np.array(tensors)
print(arrays.shape) # (30, 768)
data = arrays

def KDE():
    kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(data)
    density = kde.score_samples(data)
    
    low_density_threshold = np.percentile(density, 20)
    low_density_points = data[density < low_density_threshold] # not used currently
    
    num_samples = 100
    new_samples = kde.sample(num_samples)
    
    print(f"density: {density}")
    print(f"shape of samples: {new_samples.shape}")
    
    return new_samples

def Gaussian(num_samples=10):
    mean = np.mean(data, axis=0)  
    covariance_matrix = np.cov(data, rowvar=False)  # rowvar=False 表示每一列是一个变量  

    epsilon = 1e-6
    covariance_matrix += epsilon * np.eye(covariance_matrix.shape[0])
    multivariate_gaussian = multivariate_normal(mean=mean, cov=covariance_matrix)  

    samples = multivariate_gaussian.rvs(num_samples, random_state=42)  

    print(samples.shape)
    return samples

def GMM(num_samples=10):
    from sklearn.mixture import GaussianMixture
    import time
    
    # 创建高斯混合模型，假设使用 10 个成分
    begin = time.time()
    n_components = 3
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    
    gmm.fit(data)
    
    samples, _ = gmm.sample(num_samples)
    
    print(samples.shape)
    print(f"GMM spend: {time.time()-begin}")
    return samples

num_samples = 400
# new_samples = Gaussian(num_samples)
new_samples = GMM(num_samples)

new_samples = torch.tensor(new_samples)

idx = 196
sample_embed = new_samples[idx:idx+1]
# sample_embed = torch.tensor(tensors[4:5])
print(f"sample_embed: {sample_embed.shape}") # torch.Size([1, 768])

prompt = "a high quality photo of a holder {}.".format(class_name.replace("_", " "))
neg_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

text_inputs = text_tokenizer(prompt, return_tensors="pt", max_length=77, padding="max_length", truncation=True)["input_ids"]
tokens = text_tokenizer.convert_ids_to_tokens(text_inputs[0])
word_index = tokens.index("holder</w>")
# print(f"tokens: {tokens}")
# print(f"index: {word_index}")
text_outputs = text_model(input_ids=text_inputs.to(device))
text_embeds = text_outputs.last_hidden_state
text_embeds[:, word_index, :] = sample_embed

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

combined_image.save('temp_imgs/samples_generate.png')
# images[0].save(infer_test.png)
