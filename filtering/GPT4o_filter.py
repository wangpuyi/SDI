import os
import openai
import base64
from mimetypes import guess_type
import pandas as pd
import csv
from tqdm import tqdm

openai.api_type = "azure"
openai.api_version = "2024-02-01" 
openai.api_base = 'https://cs-icdevai02-openai-zhxuv-swc.openai.azure.com/' # Your Azure OpenAI resource's endpoint value.
openai.api_key = '9fcbba8618ff4b949ce5a98afefd4ec3'

def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    
    with open(image_path, 'rb') as f:
        base64_data = base64.b64encode(f.read()).decode('utf-8')
    
    return f"data:{mime_type};base64,{base64_data}"

def GPT4o_filter(concatenated_root, csv_path):
    img_paths = os.listdir(concatenated_root)
    for image_path in tqdm(img_paths[:5]):
        # image_path = '/mnt/hwfile/gveval/yangshuo/dataset/MoMA/CUB2011/sample_act3_withClass_5to1/288_concatenated.jpg'
        image_path = os.path.join(concatenated_root, image_path)
        img_index = int(image_path.split('/')[-1].split('_')[0])
        print(f'img_index: {img_index}')

        image_data = local_image_to_data_url(image_path)
        # print("Data URL:", image_data)

        sentence = sentences[img_index]

        response = openai.ChatCompletion.create(
            engine="gpt-4o", # The deployment name you chose when you deployed the GPT-3.5-Turbo or GPT-4 model.
            messages=[
                { "role": "system", "content": "You are a expert in bird." },
                { "role": "user", "content": [  
                    { 
                        "type": "text", 
                        "text": f"The image provided is a collection of five images. Determine which image most accurately represents {sentence}. Please specify the image explicitly in your response, using terms like 'first', 'second', 'third', 'fourth', or 'fifth'. Count from the left. In your assessment, prioritize the species and shape of the bird over its action."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data
                        }
                    }
                ] } 
            ],
        )

        print(response)
        # print(response['choices'][0]['message']['content'])
        with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            content = response['choices'][0]['message']['content'].replace('\n', ' ')
            writer.writerow([str(img_index).zfill(4), content])


if __name__ == '__main__':
    filter_save_dir = "filtering_results"
    name = 'Cub2011-1out5-setting1-MoMA-Withclass-GPT4o'
    if not os.path.exists(f"{filter_save_dir}/{name}"):
        os.makedirs(f"{filter_save_dir}/{name}")
        os.makedirs(f"{filter_save_dir}/{name}/samples")
        os.makedirs(f"{filter_save_dir}/{name}/filtered_idxs")

    sentences_file = pd.read_csv('/mnt/petrelfs/yangshuo/MoMA/output/CUB2011/CUB2011_3act_withClass.csv')
    disorder_sentences = sentences_file['sentence'].tolist()
    index = sentences_file['index'].tolist()
    sorted_pairs = sorted(zip(index, disorder_sentences))
    sentences = [sentence for _, sentence in sorted_pairs]
    concatenated_root = '/mnt/hwfile/gveval/yangshuo/dataset/MoMA/CUB2011/sample_act3_withClass_5to1'

    csv_path = f"{filter_save_dir}/{name}/GPT4o.csv"
    # with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["index", "GPT4o"])
    
    GPT4o_filter(concatenated_root, csv_path)

