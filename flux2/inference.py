import os
import torch
import pandas as pd
from tqdm import tqdm
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16, device_map="cuda")

output_dir = "prompt_test_512/w_id_15prompt"
os.makedirs(output_dir, exist_ok=True)

# csv_path = "/media/ee303/4TB/SoftREPA/tools/final_prompt_test_result_all_pose.csv"
csv_path = '/media/ee303/4TB/SoftREPA/tools/15prompt_id.csv'
df = pd.read_csv(csv_path)

for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = f'{row["prompt"]}'
    img_path = row["image_path"]
    
    if not os.path.exists(img_path):
        print(f"Skipping {img_path} - file not found.")
        continue
        
    input_image = load_image(img_path)
    image = pipe(
            image=input_image,
            prompt=prompt,
            height=512,
            width=512     
        ).images[0]
    
    basename = os.path.basename(img_path).split('.')[0]
    safe_prompt = prompt.replace(" ", "_").replace("/", "").replace(",", "")[:100]
    out_path = os.path.join(output_dir, f"{basename}_{safe_prompt}.png")
    
    image.save(out_path)