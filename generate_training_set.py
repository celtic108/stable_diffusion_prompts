from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import argparse
from prompt_vae import loader
import os

model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

text_data = loader.load_data()

def main(count):
    for i, row in text_data.iterrows():
        image_path = f"data/images/generated_images/{i}.png"
        if not os.path.isfile(image_path):
            prompt = row['TEXT']
            print(i, prompt)
            image = pipe(prompt).images[0]
            image.save(image_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'GenerateTestData',
                    description = 'Creates pairs of images and the prompts that created them',
                    epilog = 'For Kaggle Competition')
    parser.add_argument('-c', '--count', default=5)
    parser.add_argument('-p', '--path', default='default')
    args = parser.parse_args()
    main(args.count)