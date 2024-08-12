from django.shortcuts import render, redirect
from .forms import PhotoForm
from .models import Photo
import io
import os
import json
import torch
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from diffusers import StableDiffusionPipeline
from PIL import Image
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
    return render(request, 'main.html')

def llm_model(request):
    return render(request, 'llm_model.html')

def text_to_image_model(request):
    return render(request, 'text-to-image.html')

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

def pdf_converter(request):
    return render(request, 'pdf_converter.html')

def story_maker_model(request):
    return render(request, 'story_maker_model.html')

def medical_assistant_func(request):
    return render(request, 'medical_assistant.html')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# runwayml/stable-diffusion-v1-5
# dreamlike-art/dreamlike-diffusion-1.0
# Load the model once

model_path = "C:/Mehmet Genel/Mehmet Yazılım/ChatGPTurk/Chatbot/Python/Models/Models_local/local_dreamlike_diffusion"
text_to_image = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
text_to_image = text_to_image.to("cuda")

medical_assistant = AutoModelForSeq2SeqLM.from_pretrained('C:/Mehmet Genel/Mehmet Yazılım/ChatGPTurk/Chatbot/Python/Models/Models_local/local_medical_assistant_model')
medical_tokenizer = AutoTokenizer.from_pretrained('C:/Mehmet Genel/Mehmet Yazılım/ChatGPTurk/Chatbot/Python/Models/Models_local/local_medical_assistant_model')

medical_assistant.to("cpu")

def generate_image(prompt):
    image = text_to_image(prompt).images[0]
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

@csrf_exempt
def chat_response(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        input_text = data.get('user_input')

        # Tokenize edin
        inputs = medical_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

        with torch.no_grad():
            outputs = medical_assistant.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=512,       # Maksimum yanıt uzunluğu
                num_beams=2,          # Beam search
                early_stopping=True,  # Erken durdurma
                no_repeat_ngram_size=1 # Tekrarları önlemek için
            )

        generated_text = medical_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = {
            'response': generated_text
        }

        return JsonResponse(response)

from transformers import BlipProcessor, BlipForConditionalGeneration
import fitz

import io
import json
from PIL import Image
import fitz  # PyMuPDF
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BlipProcessor, BlipForConditionalGeneration

# Yüklenen dosyaların kaydedileceği dizin
UPLOAD_DIR = 'uploaded_files/'
base_path = "C:/Mehmet Genel/Mehmet Yazılım/ChatGPTurk/Chatbot/Python/Models/Models_local"

tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-tatoeba-en-tr')
model_translate = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-tatoeba-en-tr')

# Load models and tokenizer
# tokenizer = AutoTokenizer.from_pretrained(os.path.join(base_path, 'local_translate'))
# model_translate = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(base_path, 'local_translate'))

processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large')
model_pdf = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large')

# processor = BlipProcessor.from_pretrained(os.path.join(base_path, 'local_pdf_reader'))
# model_pdf = BlipForConditionalGeneration.from_pretrained(os.path.join(base_path, 'local_pdf_reader')).to("cuda")

def split_text(text, max_length=1024):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def translate_text(text, tokenizer, model):
    translated_text = ""
    chunks = split_text(text)
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        outputs = model.generate(inputs["input_ids"], max_length=1024)
        translated_chunk = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated_text += translated_chunk + " "
    return translated_text

def extract_text_and_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    images = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()

        img_list = page.get_images(full=True)
        for img_index, img in enumerate(img_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append((image, page_num))

    doc.close()
    return text, images

def save_images(images, output_dir="output_images"):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (image, page_num) in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{page_num}_image_{i}.png")
        image.save(image_path)

@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        if 'file' in request.FILES:
            pdf_file = request.FILES['file']
            file_path = default_storage.save(UPLOAD_DIR + pdf_file.name, pdf_file)
            file_full_path = 'media/' + file_path
            text, images = extract_text_and_images_from_pdf(file_full_path)
            translated_text = translate_text(text, tokenizer, model_translate)
            
            save_images(images)
            captions = generate_captions_for_images(images, processor, model_pdf)

            # Prepare JSON response
            result = {
                'text': translated_text,
                'captions': captions,
            }
            return JsonResponse(result)
        else:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)

def generate_captions_for_images(images, processor, model):
    captions = []
    for i, (image, _) in enumerate(images):
        inputs = processor(image, return_tensors="pt").to("cpu")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)
    return captions

def image_responser(request):
    return render(request, "image_responser.html")

from transformers import ViltProcessor, ViltForQuestionAnswering

processor_image = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model_image = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

@csrf_exempt
def upload_photo(request):
    if request.method == 'POST' and request.FILES.get('file') and request.POST.get('question'):
        file = request.FILES['file']
        question = request.POST.get('question')
        
        # Convert the image to RGB format
        image = Image.open(file).convert("RGB")
        
        # Process the image and question
        encoding = processor_image(image, text=question, return_tensors="pt")
        outputs = model_image(**encoding)
        
        # Extract logits and find the index with the highest logit
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        
        # Map the index to the corresponding label
        answer = model_image.config.id2label[idx]
        
        # Return the answer as plain text
        return HttpResponse(answer, content_type='text/plain')

    return HttpResponse('Invalid request', status=400)

def news(request):
    return render(request, 'news.html')

from transformers import pipeline
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

cls = pipeline("zero-shot-classification", model='facebook/bart-large-mnli')

from django.template.loader import render_to_string

@csrf_exempt
def load_news(request):
    labels = ["sport", "politics", "technology", "health", "education"]

    url = "https://www.bbc.com/news"

    sport_news_urls = []
    politics_news_urls = []
    tech_news_urls = []
    health_news_urls = []
    education_news_urls = []

    # Web sayfasını indir
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Haber başlıklarını ve URL'lerini içeren HTML elementlerini bul
    articles = soup.select('a.sc-2e6baa30-0')

    # Başlıkları ve URL'leri listeye ekle
    for article in tqdm(articles):
        title = article.get_text()
        link = article['href']
        full_link = f"https://www.bbc.com{link}" if link.startswith('/') else link

        # Zero-shot classification modelini kullanarak haber kategorisini belirle
        result = cls(title, labels)
        top_label = result['labels'][0]
        top_score = result['scores'][0]

        if top_label == "sports" and top_score >= 0.7:
            sport_news_urls.append(full_link)
        elif top_label == "politics" and top_score >= 0.7:
            politics_news_urls.append(full_link)
        elif top_label == "technology" and top_score >= 0.7:
            tech_news_urls.append(full_link)
        elif top_label == "health" and top_score >= 0.7:
            health_news_urls.append(full_link)
        elif top_label == "education" and top_score >= 0.7:
            education_news_urls.append(full_link)

    # AJAX istekleri için HTML içeriği render et
    html = render_to_string('news_content.html', {
        'sport_news_urls': sport_news_urls,
        'politics_news_urls': politics_news_urls,
        'tech_news_urls': tech_news_urls,
        'health_news_urls': health_news_urls,
        'education_news_urls': education_news_urls,
    })

    return JsonResponse({'html': html})