


# 🪨 StoneSizeAI

StoneSizeAI — это простой пайплайн для обучения object detection модели [DETR (DEtection TRansformer)](https://huggingface.co/facebook/detr-resnet-50) от Hugging Face на ваших изображениях с разметкой в формате YOLO. Подходит для обучения модели, распознающей камни или другие объекты **по изображениям без указания физического размера**.

---

## 📁 Структура проекта

```

StoneSizeAI/
├── dataset/
│   ├── images/
│   │   ├── train/      # обучающие изображения
│   │   └── val/        # валидационные изображения
│   ├── labels/
│   │   ├── train/      # YOLO-разметка к train
│   │   └── val/        # YOLO-разметка к val
├── main.py             # запуск обучения
├── train.py            # логика обучения
├── utils.py            # вспомогательные функции
├── requirements.txt    # зависимости
├── README.md           # документация

```

---

## 🖼 Подготовка данных

### Изображения

- Сложите обучающие изображения в `dataset/images/train/`
- Валидационные — в `dataset/images/val/`

Формат: `.jpg`, `.png`

### Разметка

Файлы разметки должны находиться в `dataset/labels/train/` и `dataset/labels/val/`.  
Формат разметки — **YOLO** (только 1 класс, `0`):

**Пример файла `0001.txt`:**
```

0 0.5 0.5 0.2 0.3

````

Формат:  
`<class> <x_center> <y_center> <width> <height>`  
Значения **нормализованы** (от 0 до 1). Названия файлов .txt должны совпадать с .jpg.

---

## 🚀 Обучение (на Google Colab или локально)

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
````

### 2. Запуск обучения

```bash
python main.py
```

Внутри будет происходить:

* Загрузка изображений и аннотаций
* Инициализация модели DETR
* Обучение и сохранение модели

---

## 🔮 В будущем

⚙️ Добавим:

* Inference-режим (предсказания на новых изображениях)
* Поддержка нескольких классов

---

## 📦 Зависимости

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

## 📄 Лицензия

MIT License

---

## 📬 Контакты

Создано для экспериментов и обучения.
Пишите, если нужна помощь или доработка ✉️
Автор - Akitik
