from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import random
torch.manual_seed(1234)
random.seed(0)

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

def Qwenfilter5to1(text):
    mapping = {
        'first': 0,
        'second': 1,
        'third': 2,
        'fourth': 3,
        'fifth': 4
    }
    
    for word, number in mapping.items():
        if word in text:
            return number
    
    return random.randint(0, 4)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

ans = ['first', 'second', 'third', 'fourth', 'fifth']
sentence = 'a photo of A Black footed Albatross bird stands amongst reeds and water lilies, its reflection mirrored in the still water.'
# 1st dialogue turn
query = tokenizer.from_list_format([
    {'image': '/mnt/hwfile/gveval/yangshuo/dataset/MoMA/CUB2011/sample_act3_withClass_5to1/20_concatenated.jpg'}, # Either a local path or an url
    {'text': f"Examine the collection of five images provided. Identify whether the first, second, third, fourth, or fifth image from the left most accurately depicts {sentence}.\
      In your assessment, prioritize the accuracy of the bird's species and shape over the specifics of its action. Provide a rationale for your choice."},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
print(Qwenfilter5to1(response)) 