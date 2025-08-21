import torch
from transformers import DetrForObjectDetection, DetrImageProcessor, AutoConfig
import argparse
import os

# --- Настройки ---
checkpoint_path = "./detr/output/checkpoint.pth"  # путь к твоему PyTorch чекпойнту
output_dir = "./detr/hf_model"                    # папка для HuggingFace модели
base_model_name = "facebook/detr-resnet-50"       # для инициализации HF модели

os.makedirs(output_dir, exist_ok=True)

# --- Разрешаем безопасно загрузить Namespace из checkpoint ---
torch.serialization.add_safe_globals([argparse.Namespace])

# --- Загружаем checkpoint с весами ---
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
print("Keys in checkpoint:", checkpoint.keys())

# --- Создаём модель HuggingFace и заливаем веса ---
print("Loading base HF DETR model...")
model = DetrForObjectDetection.from_pretrained(base_model_name)

# Заменяем веса модели на свои
model.load_state_dict(checkpoint["model"], strict=False)
print("Weights loaded into HF model.")

# --- Создаём и сохраняем processor ---
processor = DetrImageProcessor.from_pretrained(base_model_name)
processor.save_pretrained(output_dir)
print(f"Processor saved to {output_dir}")

# --- Сохраняем модель HuggingFace ---
model.save_pretrained(output_dir)
print(f"HF model saved to {output_dir}")

print("✅ Конвертация завершена! Теперь можно запускать:")
print(f"processor = DetrImageProcessor.from_pretrained('{output_dir}')")
print(f"model = DetrForObjectDetection.from_pretrained('{output_dir}')")
