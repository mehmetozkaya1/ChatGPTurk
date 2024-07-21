from django.shortcuts import render, redirect
from .forms import PhotoForm
from .models import Photo

def upload_photo(request):
    if request.method == 'POST':
        form = PhotoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('photo_list')
    else:
        form = PhotoForm()
    return render(request, 'upload_photo.html', {'form': form})

def photo_list(request):
    photos = Photo.objects.all()
    return render(request, 'photo_list.html', {'photos': photos})

def index(request):
    return render(request, 'chatbot.html')

def llm_model(request):
    return render(request, 'llm_model.html')

def text_to_image_model(request):
    return render(request, 'text_to_image_model.html')

def web_scraping_model(request):
    return render(request, 'web_scraping_model.html')

def image_commenter_model(request):
    if request.method == 'POST':
        form = PhotoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('photo_list')
    else:
        form = PhotoForm()
    return render(request, 'image_commenter_model.html', {'form': form})

def story_maker_model(request):
    return render(request, 'story_maker_model.html')

def models(request):
    return render(request, 'models.html')

import io
import os
import json
import torch
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from diffusers import StableDiffusionPipeline
from PIL import Image

# Set environment variable to avoid OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the model once
model_id = "dreamlike-art/dreamlike-diffusion-1.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def generate_image(prompt):
    image = pipe(prompt).images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf

@csrf_exempt
def generate_image_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            prompt = data.get('prompt', '')
            if not prompt:
                return JsonResponse({'error': 'Prompt is required'}, status=400)
            
            # Generate the image
            image_buf = generate_image(prompt)
            
            response = HttpResponse(image_buf, content_type='image/png')
            response['Content-Disposition'] = 'attachment; filename="generated_image.png"'
            return response
        
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

    return JsonResponse({'error': 'POST request required'}, status=400)
