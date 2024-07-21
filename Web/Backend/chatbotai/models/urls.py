from django.urls import path
from . import views

urlpatterns = [
    path('photos/', views.photo_list, name='photo_list'),
    path('', views.index, name="index"),
    path('models/', views.models, name="models"),
    path('models/llm-model', views.llm_model, name="llm-model"),
    path('models/text-to-image-model', views.text_to_image_model, name="text-to-image-model"),
    path('models/web-scraping-model', views.web_scraping_model, name="web-scraping-model"),
    path('models/image-commenter-model', views.image_commenter_model, name="image-commenter-model"),
    path('models/story-maker-model', views.story_maker_model, name="story-maker-model"),
    path('generate-image/', views.generate_image_view, name='generate_image'),
]
