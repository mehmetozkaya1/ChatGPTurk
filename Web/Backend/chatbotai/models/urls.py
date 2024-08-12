from django.urls import path
from . import views

urlpatterns = [
    path('photos/', views.photo_list, name='photo_list'),
    path('', views.index, name="index"),
    path('models/llm-model', views.llm_model, name="llm-model"),
    path('models/text-to-image-model', views.text_to_image_model, name="text-to-image-model"),
    path('models/web-scraping-model', views.web_scraping_model, name="web-scraping-model"),
    path('models/image-commenter-model', views.image_commenter_model, name="image-commenter-model"),
    path('models/story-maker-model', views.story_maker_model, name="story-maker-model"),
    path('models/medical-assistant-model', views.medical_assistant_func, name="medical-assistant"),
    path('chat_response/', views.chat_response, name='chat_response'),
    path('generate_image_view/', views.generate_image_view, name='generate_image_view'),
    path('models/pdf-converter-model', views.pdf_converter, name="pdf-converter-model"),
    path('upload/', views.upload_file, name='upload_file'),
    path('models/image-responser-model', views.image_responser, name="image-responser"),
    path('upload_photo/', views.upload_photo, name='upload_photo'),
    path('models/news', views.news, name="news-nodel"),
    path('news/', views.load_news, name='load_news')
]
