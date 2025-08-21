"""
main.py

Функции:
 - convert  : конвертирует YOLO .txt -> COCO JSON (instances_train2017.json)
 - create-empty : создаёт пустой COCO (images + пустые annotations) (для отладки)
 - infer    : запускает real-time инференс DETR (Hugging Face) с ArUco калибровкой mm/px

Пример запуска:
 python main.py convert
 python main.py create-empty
 python main.py infer --model facebook/detr-resnet-50 --video 0 --calib calib/calib_aruco.jpg

Примечание: Для тренировки DETR рекомендую использовать Hugging Face Cookbook / ноутбук (см. README в проекте).
"""

import os
import json
import argparse
import time
from PIL import Image
import numpy as np
import cv2
import torch

# Transformers imports
from transformers import DetrImageProcessor, DetrForObjectDetection, get_linear_schedule_with_warmup
from torch.optim import AdamW

# ---------------------------
#  Настройки (измените под себя)
# ---------------------------
DATASET_ROOT = "dataset"
IMAGES_TRAIN = os.path.join(DATASET_ROOT, "train2017")
IMAGES_VAL = os.path.join(DATASET_ROOT, "val2017")
LABELS_TRAIN = os.path.join(DATASET_ROOT, "labels", "train")
LABELS_VAL = os.path.join(DATASET_ROOT, "labels", "val")
ANNOTATIONS_DIR = os.path.join(DATASET_ROOT, "annotations")
TRAIN_JSON = os.path.join(ANNOTATIONS_DIR, "instances_train2017.json")
VAL_JSON = os.path.join(ANNOTATIONS_DIR, "instances_val2017.json")
CLASSES_PATH = "classes.txt"  # если есть

ARUCO_MARKER_MM = 100.0  # физический размер маркера (мм), измерьте и пропишите правильное значение
MM_THRESHOLD = 300.0     # порог в мм для oversized
STOP_THROTTLE = 5.0      # сек между остановками (чтобы не гонять стоп постоянно)

# ---------------------------
#  Утилиты: YOLO -> COCO
# ---------------------------
def read_classes(classes_path):
    if not os.path.exists(classes_path):
        return ["rock", "foreign_object"]
    return [x.strip() for x in open(classes_path, "r", encoding="utf-8").read().splitlines() if x.strip()]


# ... (остальной код без изменений)

# Добавим константы для info и licenses
COCO_INFO = {
    "year": 2023,
    "version": "1.0",
    "description": "Stone Size Dataset",
    "contributor": "Your Name",
    "date_created": "2023-01-01"
}

COCO_LICENSES = [
    {
        "id": 1,
        "name": "CC BY-NC-SA 2.0",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]


def yolo_to_coco(images_dir, labels_dir, classes_path, out_json_path):
    """
    Преобразует YOLO-разметку (txt) в COCO JSON.
    images_dir: папка с изображениями
    labels_dir: папка с .txt (один файл на изображение, если нет - считается, что объект не размечён)
    classes_path: файл classes.txt
    """
    classes = read_classes(classes_path)
    images = []
    annotations = []
    ann_id = 1
    img_id = 1
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)

    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(images_dir, fname)
        try:
            w, h = Image.open(img_path).size
        except Exception as e:
            print("Cannot open image:", img_path, e);
            continue

        images.append({"file_name": fname, "height": h, "width": w, "id": img_id})
        label_file = os.path.join(labels_dir, os.path.splitext(fname)[0] + '.txt')
        if os.path.exists(label_file):
            for line in open(label_file, 'r', encoding='utf-8'):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w_rel = float(parts[3])
                h_rel = float(parts[4])
                x = (x_center - w_rel / 2) * w
                y = (y_center - h_rel / 2) * h
                bw = w_rel * w
                bh = h_rel * h
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls,
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0
                })
                ann_id += 1
        img_id += 1

    categories = [{"id": i+1, "name": name} for i, name in enumerate(classes)]

    # Добавляем info и licenses в COCO JSON
    coco = {
        "info": COCO_INFO,
        "licenses": COCO_LICENSES,
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    print("Saved COCO annotations:", out_json_path)
    print("Images:", len(images), "Annotations:", len(annotations), "Classes:", len(categories))


def create_empty_coco(images_dir, out_json_path):
    """
    Создает COCO JSON со списком изображений, но пустыми аннотациями.
    Удобно для отладки pipeline без реальных аннотаций.
    """
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    images = []
    img_id = 1
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')): continue
        w, h = Image.open(os.path.join(images_dir, fname)).size
        images.append({"file_name": fname, "height": h, "width": w, "id": img_id})
        img_id += 1
    categories = [{"id": 0, "name": "rock"}, {"id": 1, "name": "foreign_object"}]

    # Добавляем info и licenses в COCO JSON
    coco = {
        "info": COCO_INFO,
        "licenses": COCO_LICENSES,
        "images": images,
        "annotations": [],
        "categories": categories
    }

    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    print("Created empty COCO:", out_json_path, "images:", len(images))


# ... (остальной код без изменений)


# ---------------------------
#  ArUco calibration helper
# ---------------------------
def compute_mm_per_px_from_aruco(image_path, marker_mm=ARUCO_MARKER_MM, aruco_dict_type=cv2.aruco.DICT_4X4_50):
    img = cv2.imread(image_path)
    if img is None:
        print("[CALIB] Cannot read", image_path); return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        print("[CALIB] ArUco not found in calibration image.")
        return None
    c = corners[0][0]
    side_lengths = [np.linalg.norm(c[i] - c[(i+1) % 4]) for i in range(4)]
    avg_px = float(np.mean(side_lengths))
    mm_per_px = marker_mm / avg_px
    print(f"[CALIB] avg_px={avg_px:.2f} -> mm_per_px={mm_per_px:.6f}")
    return mm_per_px

# ---------------------------
#  Inference (DETR)
# ---------------------------
def run_inference(model_name_or_path, video_source=0, calib_image=None, conf_thr=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Loading model:", model_name_or_path, "device:", device)
    processor = DetrImageProcessor.from_pretrained(model_name_or_path)
    model = DetrForObjectDetection.from_pretrained(model_name_or_path).to(device)
    model.eval()

    mm_per_px = None
    if calib_image and os.path.exists(calib_image):
        mm_per_px = compute_mm_per_px_from_aruco(calib_image)
    if mm_per_px is None:
        # попросим вручную
        try:
            mm_per_px = float(input("[INPUT] Введите mm_per_px вручную (например 0.5): "))
        except Exception:
            mm_per_px = None
            print("[WARN] mm_per_px не установлен. Оценка физических размеров будет недоступна.")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("[ERROR] Не удалось открыть видео-источник:", video_source)
        return

    last_alert = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Конец потока / нет кадра"); break

        inputs = processor(images=frame, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([frame.shape[:2]], device=device)
        results = processor.post_process(outputs, target_sizes=target_sizes)[0]

        annotated = frame.copy()
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            score = float(score)
            if score < conf_thr:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            w_px = x2 - x1
            h_px = y2 - y1
            size_mm = None
            if mm_per_px is not None:
                size_mm = max(w_px, h_px) * mm_per_px
            cls_name = model.config.id2label[int(label)] if int(label) in model.config.id2label else str(int(label))
            color = (0, 255, 0)
            if cls_name == "foreign_object" or (size_mm is not None and size_mm > MM_THRESHOLD):
                color = (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"{cls_name} {score:.2f}"
            if size_mm is not None:
                text += f" {int(size_mm)}mm"
            cv2.putText(annotated, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if (cls_name == "foreign_object" or (size_mm is not None and size_mm > MM_THRESHOLD)) and (time.time() - last_alert > STOP_THROTTLE):
                print(f"[ALERT] {cls_name} detected, size={size_mm}, score={score:.2f}")
                print("[ACTION] Emulating conveyor STOP (replace with serial/modbus call)")
                last_alert = time.time()

        cv2.imshow("Conveyor Monitor", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("[MANUAL] Manual stop requested.")
            print("[ACTION] Emulating conveyor STOP")

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------
#  CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", help="sub-command")

    p_conv = sub.add_parser("convert", help="Convert YOLO -> COCO for train and val")
    p_conv.add_argument("--train_images", default=IMAGES_TRAIN, help="train images dir")
    p_conv.add_argument("--train_labels", default=LABELS_TRAIN, help="train labels dir")
    p_conv.add_argument("--val_images", default=IMAGES_VAL, help="val images dir")
    p_conv.add_argument("--val_labels", default=LABELS_VAL, help="val labels dir")
    p_conv.add_argument("--train_out", default=TRAIN_JSON, help="out coco json for train")
    p_conv.add_argument("--val_out", default=VAL_JSON, help="out coco json for val")
    p_conv.add_argument("--classes", default=CLASSES_PATH, help="classes.txt")

    p_empty = sub.add_parser("create-empty", help="create empty COCO with images listed")
    p_empty.add_argument("--images", default=IMAGES_TRAIN)
    p_empty.add_argument("--out", default=TRAIN_JSON)

    p_inf = sub.add_parser("infer", help="Run realtime inference (DETR)")
    p_inf.add_argument("--model", required=True, help="model name or path (e.g. facebook/detr-resnet-50 or path to checkpoint)")
    p_inf.add_argument("--video", default=0, help="video source: 0 or path or rtsp")
    p_inf.add_argument("--calib", default=os.path.join("calib", "calib_aruco.jpg"), help="calibration image with ArUco marker")
    p_inf.add_argument("--conf", default=0.5, type=float, help="confidence threshold")

    args = parser.parse_args()

    if args.cmd == "convert":
        print("Converting YOLO -> COCO for train")
        yolo_to_coco(args.train_images, args.train_labels, args.classes, args.train_out)
        print("Converting YOLO -> COCO for val")
        yolo_to_coco(args.val_images, args.val_labels, args.classes, args.val_out)
    elif args.cmd == "create-empty":
        print("Create empty COCO")
        create_empty_coco(args.images, args.out)
    elif args.cmd == "infer":
        run_inference(args.model, args.video, args.calib, args.conf)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
