import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers import StableDiffusionPipeline

from ip_adapter.ip_adapter import SEIImageProjModel
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets

import subprocess
import wandb
from datetime import datetime

def setup_ddp(rank, world_size, args):
    master_node = subprocess.getoutput("scontrol show hostname $SLURM_JOB_NODELIST | head -n 1")
    os.environ['MASTER_ADDR'] = master_node
    os.environ['MASTER_PORT'] = str(args.port)
    print("Master address: {}".format(os.environ['MASTER_ADDR']))
    print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
    init_process_group(backend='nccl', rank=rank, world_size=world_size)


# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, root, tokenizer, size=512, i_drop_rate=0.05):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate

        dataset = datasets.ImageFolder(root)

        self.data = dataset.samples # list of dict: [{"image_file": "1.png"}]
        print(f"len of dataset: {len(self.data)}")

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()
        
    def __getitem__(self, idx):
        image_file_path, label = self.data[idx]

        # read image
        raw_image = Image.open(image_file_path)
        image = self.transform(raw_image.convert("RGB"))
        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        
        prompt = "a high quality photo of a class."
        text_input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"] # tensor([[49406, 40293, 49407, ...]])
        tokens = self.tokenizer.convert_ids_to_tokens(text_input_ids[0])
        word_index = tokens.index("class</w>")

        return {
            "image": image,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "label": int(label),
            "text_input_ids": text_input_ids,
            "word_index": word_index
        }

    def __len__(self):
        return len(self.data)
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    labels = [example["label"] for example in data]
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    word_indices = [example["word_index"] for example in data]

    return {
        "images": images,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
        "labels": labels,
        "text_input_ids": text_input_ids,
        "word_indices": word_indices
    }
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--clip_path",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        # required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        # required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--port", type=str, required=True, help="Port number for distributed training")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main(rank, world_size):
    print(f"rank: {rank}")
    args = parse_args()
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Start Time: {current_time}")

    setup_ddp(rank, world_size, args)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        print(f"run on cuda:{rank}")
    else:
        device = torch.device("cpu")
        print("run on cpu")

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    weight_dtype = torch.float32

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.clip_path)
    pipe = StableDiffusionPipeline.from_pretrained(
        "/grp01/cs_hszhao/cs002u03/ckpt/stable-diffusion-v1-5",
        torch_dtype=weight_dtype,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    ).to(device)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    # freeze parameters of models to save more memory
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    image_encoder.requires_grad_(False)
    # pipe.requires_grad_(False)
    # text_encoder.requires_grad_(False)

    vae.to(device, dtype=weight_dtype)
    image_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    # noise_scheduler.to(device, dtype=weight_dtype)
    
    image_proj_model = SEIImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
    )
    
    image_proj_model = DDP(image_proj_model.to(rank), device_ids=[rank])

    # optimizer
    params_to_opt = image_proj_model.module.parameters()
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = MyDataset(args.data_path, tokenizer=tokenizer, size=args.resolution)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        sampler=train_sampler,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        train_sampler.set_epoch(epoch)
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            # print(f"global_step: {global_step}")
            load_data_time = time.perf_counter() - begin
            # Convert images to latent space
            with torch.no_grad():
                latents = vae.encode(batch["images"].to(device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
            with torch.no_grad():
                image_embeds = image_encoder(batch["clip_images"].to(device, dtype=weight_dtype)).image_embeds
            image_embeds_ = []
            for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                if drop_image_embed == 1:
                    image_embeds_.append(torch.zeros_like(image_embed))
                else:
                    image_embeds_.append(image_embed)
            image_embeds = torch.stack(image_embeds_)
            # print(f"image_embeds shape: {image_embeds.shape}") # torch.Size([32, 768])
            projected_image_embeds = image_proj_model(image_embeds)
            # print(f"projected_image_embeds shape: {projected_image_embeds.shape}") # torch.Size([32, 768])

            # shape (bs, 77, hidden_space)
            text_embeddings = text_encoder(input_ids=batch["text_input_ids"].to(device)).last_hidden_state
            text_embeddings = text_embeddings.to(dtype=weight_dtype)
            for i, (word_index, projected_image_embed) in enumerate(zip(batch["word_indices"], projected_image_embeds)):
                text_embeddings[i, word_index, :] = projected_image_embed
                    
            # print(f"noisy_latents dtype: {noisy_latents.dtype}, device: {noisy_latents.device}") # torch.float32, cuda:0
            # print(f"timesteps dtype: {timesteps.dtype}, device: {timesteps.device}") # torch.int64, cuda:0
            # print(f"text_embeddings dtype: {text_embeddings.dtype}, device: {text_embeddings.device}") # torch.float32, cuda:0
            noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
                
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")


            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print("Epoch {}, step: {}, data_time: {}, time: {}, loss: {}".format(
                epoch, step, load_data_time, time.perf_counter()-begin, loss
            ))
            
            global_step += 1
            
        if rank == 0 and (epoch+1) % 10 == 0:
            save_path = os.path.join(args.output_dir+"/bs{}-lr_{}-{}".format(args.train_batch_size*world_size, 
                                                                        args.learning_rate, current_time),
                                        f"checkpoint-{epoch}.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(image_proj_model.module.state_dict(), save_path)
        begin = time.perf_counter()

    destroy_process_group()
                
if __name__ == "__main__":
    import sys
    world_size = torch.cuda.device_count()
    print(f"GPU nums: {world_size}")
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)