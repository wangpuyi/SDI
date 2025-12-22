import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
import numpy as np

base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "openai/clip-vit-large-patch14"

# adapter_ckpt = "/mnt/hwfile/gveval/yangshuo/ckpt/adapter/IN100/2fc-bs32-lr_0.0001-lambda1/checkpoint-8000.pt"

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
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

# load SD pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
).to(device)

# Access the tokenizer and text encoder
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder

wnids = ['n01498041', 'n01514859', 'n01582220', 'n01608432', 'n01616318',
        'n01443537', 'n01776313', 'n01806567', 'n01833805', 'n01882714',
        'n01910747', 'n01944390', 'n01985128', 'n02007558', 'n02071294',
        'n02085620', 'n02114855', 'n02123045', 'n02128385', 'n02129165',
        'n02129604', 'n02165456', 'n02190166', 'n02219486', 'n02226429',
        'n02279972', 'n02317335', 'n02326432', 'n02342885', 'n02363005',
        'n02391049', 'n02395406', 'n02403003', 'n02422699', 'n02442845',
        'n02444819', 'n02480855', 'n02510455', 'n02640242', 'n02672831',
        'n02687172', 'n02701002', 'n02730930', 'n02769748', 'n02782093',
        'n02787622', 'n02793495', 'n02799071', 'n02802426', 'n02814860',
        'n02840245', 'n02906734', 'n02948072', 'n02980441', 'n02999410',
        'n03014705', 'n03028079', 'n03032252', 'n03125729', 'n03160309',
        'n03179701', 'n03220513', 'n03249569', 'n03291819', 'n03384352',
        'n03388043', 'n03450230', 'n03481172', 'n03594734', 'n03594945',
        'n03627232', 'n03642806', 'n03649909', 'n03661043', 'n03676483',
        'n03724870', 'n03733281', 'n03759954', 'n03761084', 'n03773504',
        'n03804744', 'n03916031', 'n03938244', 'n04004767', 'n04026417',
        'n04090263', 'n04133789', 'n04153751', 'n04296562', 'n04330267',
        'n04371774', 'n04404412', 'n04465501', 'n04485082', 'n04507155',
        'n04536866', 'n04579432', 'n04606251', 'n07714990', 'n07745940']
words = ['stingray', 'hen', 'magpie', 'kite', 'vulture',
        'goldfish',   'tick', 'quail', 'hummingbird', 'koala',
        'jellyfish', 'snail', 'crawfish', 'flamingo', 'orca',
        'chihuahua', 'coyote', 'tabby', 'leopard', 'lion',
        'tiger','ladybug', 'fly' , 'ant', 'grasshopper',
        'monarch', 'starfish', 'hare', 'hamster', 'beaver',
        'zebra', 'pig', 'ox', 'impala',  'mink',
        'otter', 'gorilla', 'panda', 'sturgeon', 'accordion',
        'carrier', 'ambulance', 'apron', 'backpack', 'balloon',
        'banjo','barn','baseball', 'basketball', 'beacon',
        'binder', 'broom', 'candle', 'castle', 'chain',
        'chest', 'church', 'cinema', 'cradle', 'dam',
        'desk', 'dome', 'drum','envelope', 'forklift',
        'fountain', 'gown', 'hammer','jean', 'jeep',
        'knot', 'laptop', 'mower', 'library','lipstick',
        'mask', 'maze', 'microphone','microwave','missile',
        'nail', 'perfume','pillow','printer','purse',
        'rifle', 'sandal', 'screw','stage','stove',
        'swing','television','tractor','tripod','umbrella',
        'violin','whistle','wreck', 'broccoli', 'strawberry'
        ]
print(f"length of words: {len(words)}")
words = [item[1] for item in sorted(zip(wnids, words))]

words_embeds = []
for i, word in enumerate(words):
    prompt = f"a real photo of a {word}."

    # Tokenize the positive prompt
    text_inputs = tokenizer(prompt, return_tensors="pt", max_length=77, padding="max_length", truncation=True)
    input_ids = text_inputs["input_ids"].to(device)
    with torch.no_grad():
        text_outputs = text_encoder(input_ids=input_ids)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    if f"{word}</w>" in tokens:
        word_index = tokens.index(f"{word}</w>")
        word_embed = text_outputs.last_hidden_state[0, word_index, :].cpu().numpy()

        words_embeds.append(word_embed)
    else:
        print(f"Token not found for word: {word}")
        raise NotImplementedError("Token not successfully found.")
words_embeds = np.stack(words_embeds)
print(words_embeds.shape)
np.save("IN100.npy", words_embeds)
