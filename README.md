


# 🪨 StoneSizeAI

StoneSizeAI — это простой пайплайн для обучения object detection модели [DETR (DEtection TRansformer)](https://huggingface.co/facebook/detr-resnet-50) от Hugging Face на ваших изображениях с разметкой в формате YOLO. Подходит для обучения модели, распознающей камни или другие объекты **по изображениям без указания физического размера**.

---

Файл `requirements.txt`:

```
torch>=1.13.0
torchvision>=0.14.0
transformers>=4.31.0
datasets>=2.14.0
opencv-python
scipy
matplotlib
pyyaml
tqdm
```

Установка:

```bash
pip install -r requirements.txt
```

---

