from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Yerel olarak kaydedilmiş modeli yükleyin
model = T5ForConditionalGeneration.from_pretrained('Chatbot/Python/Models/Models_local/local_medical_assistant_model')
tokenizer = T5Tokenizer.from_pretrained('Chatbot/Python/Models/Models_local/local_medical_assistant_model')

model.to("cpu")

# Örnek giriş
input_text = "I have a terrible headache doctor. Please explain."

# Tokenize edin
inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

# Modeli değerlendirme moduna alın
model.eval()

# Yanıt üretin
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=150,       # Maksimum yanıt uzunluğu
        num_beams=3,          # Beam search
        early_stopping=True,  # Erken durdurma
        no_repeat_ngram_size=1 # Tekrarları önlemek için
    )

# Yanıtı decode edin
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated response: {generated_text}")
