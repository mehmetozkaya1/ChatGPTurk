# from diffusers import StableDiffusionPipeline

# model_name =  "dreamlike-art/dreamlike-diffusion-1.0"

# pipeline = StableDiffusionPipeline.from_pretrained(model_name)
# pipeline.save_pretrained('Chatbot/Python/Models/Models_local/local_dreamlike_diffusion')

# model_name = 'runwayml/stable-diffusion-v1-5'

# pipeline = StableDiffusionPipeline.from_pretrained(model_name)
# pipeline.save_pretrained('Chatbot/Python/Models/Models_local/local_runwayml_diffusion')

from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

# prepare image + question
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "How many cats are there?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])