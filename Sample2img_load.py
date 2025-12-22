import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image

from ip_adapter import SEIImageProjModel
from torchvision import transforms
import os
import time
import json
from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal  
import numpy as np


base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"

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

text_tokenizer = pipe.tokenizer
text_model = pipe.text_encoder

def load_data(folder_path):
    tensors = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pt"):
            tensor = torch.load(os.path.join(folder_path, filename)).cpu()
            tensors.append(tensor.detach().numpy().flatten())
    data = np.array(tensors)
    return data

# class_name = "n03594945" # jeep
# class_name = "n01443537" # goldfish
class_name = "n01582220" # magpie
# class_name = "n04485082" # tripod
# class_name = "n02840245" # binder
# class_name = "n04507155" # umbrella

# pt_root = "/mnt/hwfile/gveval/yangshuo/output/SEI_tensor/IN100_2-sentence-14k"
# pt_root = "/mnt/hwfile/gveval/yangshuo/output/SEI_tensor/IN100_2-sentence-48k-NoAnchor"
pt_root = "/mnt/hwfile/gveval/yangshuo/output/SEI_tensor/IN100_2-sentence-48k-MSE"

arrays = load_data(os.path.join(pt_root, class_name))
print(arrays.shape)
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

prompt = "a high quality photo of a class." # class is a placeholder
neg_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

text_inputs = text_tokenizer(prompt, return_tensors="pt", max_length=77, padding="max_length", truncation=True)["input_ids"]
tokens = text_tokenizer.convert_ids_to_tokens(text_inputs[0])
word_index = tokens.index("class</w>")
# print(f"tokens: {tokens}")
# print(f"index: {word_index}")
text_outputs = text_model(input_ids=text_inputs.to(device))
text_embeds = text_outputs.last_hidden_state

neg_text_inputs = text_tokenizer(neg_prompt, return_tensors="pt", max_length=77, padding="max_length", truncation=True)["input_ids"]
neg_text_outputs = text_model(input_ids=neg_text_inputs.to(device))
neg_text_embeds = neg_text_outputs.last_hidden_state

idxs = 3
for idx in range(idxs, idxs+3):
    print(f"idx is : {idx}")
    sample_embed = new_samples[idx:idx+1]
    # sample_embed = torch.tensor(tensors[4:5])
    # print(f"sample_embed: {sample_embed.shape}") # torch.Size([1, 768])
    text_embeds[:, word_index, :] = sample_embed

    # Generate the image
    num = 5
    # images = pipe(prompt_embeds=text_embeds.repeat(num,1,1)).images
    images = pipe(prompt_embeds=text_embeds.repeat(num,1,1), negative_prompt_embeds=neg_text_embeds.repeat(num,1,1)).images

    image_arrays = [np.array(img) for img in images]
    height, width, _ = image_arrays[0].shape

    combined_image = Image.new('RGB', (width * len(images), height))

    for i, img in enumerate(images):
        combined_image.paste(img, (i * width, 0))

    combined_image.save(f'temp_imgs/samples_load_{idx}.png')
    # images[0].save(infer_test.png)
