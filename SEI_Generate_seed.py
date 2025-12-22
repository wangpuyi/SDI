import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, StableDiffusionImg2ImgPipeline
from PIL import Image
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor


from ip_adapter import SEIImageProjModel
import os
import numpy as np
import json

import sys
import logging
import random
import argparse
from datetime import datetime

def sample_func(args):
    if args.sample_func == "Gaussian":
        return Gaussian
    elif args.sample_func == "GMM3":
        return GMM3
    elif args.sample_func == "GMM5":
        return GMM5
    else:
        raise ValueError("sample_func should be either Gaussian or GMM.")

def Gaussian(data, num_samples=10):
    from scipy.stats import multivariate_normal  
    mean = np.mean(data, axis=0)  
    covariance_matrix = np.cov(data, rowvar=False)  # rowvar=False 表示每一列是一个变量  

    epsilon = 1e-6
    covariance_matrix += epsilon * np.eye(covariance_matrix.shape[0])
    multivariate_gaussian = multivariate_normal(mean=mean, cov=covariance_matrix)  

    samples = multivariate_gaussian.rvs(num_samples, random_state=42)  # return samples as numpy arrays
    samples = torch.tensor(samples)

    print(samples.shape)
    return samples

def GMM3(data, num_samples=10):
    from sklearn.mixture import GaussianMixture
    import time
    
    # 创建高斯混合模型，假设使用 10 个成分
    begin = time.time()
    n_components = 3
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    
    gmm.fit(data)

    samples, _ = gmm.sample(num_samples)
    samples = torch.tensor(samples)

    print(f"samples shape: {samples.shape}")
    print(f"GMM spend: {time.time()-begin}")
    return samples

def GMM5(data, num_samples=10):
    from sklearn.mixture import GaussianMixture
    import time
    
    # 创建高斯混合模型，假设使用 10 个成分
    begin = time.time()
    n_components = 5
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    
    gmm.fit(data)

    samples, _ = gmm.sample(num_samples)
    samples = torch.tensor(samples)

    print(f"samples shape: {samples.shape}")
    print(f"GMM spend: {time.time()-begin}")
    return samples

def load_file(wnid, args):
    return os.listdir(os.path.join(args.data_root, wnid))

def main(gpu_id, args):
    num_gpus = args.num_gpus
    # log settings
    print("project=SEI-generate", "config=", args)
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d-%H:%M:%S")

    base_model_path = "/grp01/cs_hszhao/cs002u03/ckpt/stable-diffusion-v1-5"
    vae_model_path = "/grp01/cs_hszhao/cs002u03/ckpt/sd-vae-ft-mse"

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

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
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    ).to(device)
    # image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(device)
    # clip_image_processor = CLIPImageProcessor()
    text_tokenizer = pipe.tokenizer
    text_model = pipe.text_encoder

    ###########################
    # general text embeddings #
    ###########################
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
    #################################
    # generate imgs from embeddings #
    #################################
    pt_root = args.pt_root
    categories = os.listdir(pt_root)
    length = len(categories)
    print(f"length: {length}")
        
    if num_gpus > 1 and gpu_id != num_gpus-1:
        chunk_size = length // num_gpus
        begin = chunk_size * gpu_id
        end = chunk_size * (gpu_id + 1)
    elif num_gpus > 1 and gpu_id == num_gpus-1:
        begin = length // num_gpus * gpu_id
        end = length
    else:
        begin = 0
        end = length
    print(f"Generating images for {begin} to {end}... | total length: {length}")

    for category in sorted(categories)[begin:end]:
        print(f"generating imgs for class: {category}...") # n03724870
        data = []
        image_paths = load_file(category, args)
        print(f"length of init images: {len(image_paths)}")

        pt_names = os.listdir(os.path.join(pt_root, category))
        for pt_name in pt_names:
            pt_path = os.path.join(pt_root, category, pt_name)
            data.append(torch.load(pt_path).numpy())
        data = np.stack(data) # array Size([200, 768])

        samples = sample_func(args)(data, num_samples=args.expanded_number_per_sample * len(image_paths))
        print(f"samples shape: {samples.shape} device: {samples.device}") #device: cpu

        # continue generating images
        if  os.path.exists(f"{args.save_root}/{category}") and len(os.listdir(f"{args.save_root}/{category}")) == args.expanded_number_per_sample*len(image_paths):
            print(f"skip class {category}.")
        else:
            for idx in range(len(samples)):
                text_embeds[:, word_index, :] = samples[idx].unsqueeze(0)
                # img 2 img
                image_path = os.path.join(args.data_root, category, image_paths[idx//args.expanded_number_per_sample])
                PIL_image = Image.open(image_path).convert("RGB").resize((512, 512))
                # print(f"idx: {idx}  image path: {image_path}")
                img = pipe(prompt_embeds=text_embeds, negative_prompt_embeds=neg_text_embeds, 
                        image=PIL_image, strength=args.strength, guidance_scale=args.guidance_scale).images[0]
                save_root = args.save_root
                save_path = f"{save_root}/{category}/{idx:04d}.png"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                img.save(save_path)


if __name__ == "__main__":
    import multiprocessing
    parse = argparse.ArgumentParser(description='base settings')
    # parse.add_argument('--num_gpus', type=int, required=True)
    parse.add_argument('--strength', type=float, required=True) # control the inference of text regard to image
    parse.add_argument('--guidance_scale', type=float, required=True)

    parse.add_argument('--pt_root', type=str, required=True)
    parse.add_argument('--save_root', type=str, required=True)
    parse.add_argument('--data_root', type=str, required=True)
    parse.add_argument('--expanded_number_per_sample', type=int, required=True)
    parse.add_argument('--sample_func', type=str, required=True)
    args = parse.parse_args()

    # num_gpus = args.num_gpus

    # if num_gpus is None or num_gpus <= 0:
    num_gpus = torch.cuda.device_count()
    args.num_gpus = num_gpus
    
    processes = []
    for gpu_id in range(num_gpus):
        p = multiprocessing.Process(target=main, args=(gpu_id, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()