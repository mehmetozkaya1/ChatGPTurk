import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import torch

model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"

# Load the model
pipe = StableDiffusionPipeline.from_pretrained(model_id1, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Define the prompt
prompt = """dreamlikeart, a grungy woman with rainbow hair, travelling between dimensions, dynamic pose, happy, soft eyes and narrow chin,
extreme bokeh, dainty figure, long hair straight down, torn kawaii shirt and baggy jeans
"""

# Generate the image
image = pipe(prompt).images[0]

# Print the prompt and display the image
print("[PROMPT]:", prompt)
plt.imshow(image)
plt.axis('off')
plt.show()
