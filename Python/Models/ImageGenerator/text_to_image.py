from diffusers import DiffusionPipeline
import torch
from PIL import Image
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda") 

def generate_image(prompt):
    images = pipe(prompt=prompt).images[0]
    return images

