# gui_infer.py
import cv2
import time
import threading
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Ultralytics
from ultralytics import YOLO

# ------------------ Настройки (адаптируйте) ------------------
VIDEO_SRC = 0  # USB камера index
MODEL_WEIGHTS = "runs/detect/train/weights/best.pt"  # замените, если нужно
FALLBACK_MODEL = "yolov8n.pt"  # для теста, если у вас нет обученных весов
PIXELS_PER_MM = 5.0  # <- Замените на результат вашей калибровки
SIZE_THRESHOLD_MM = 300.0
CONF_THRESHOLD = 0.35
FRAME_WIDTH = 1280  # можно уменьшить для скорости
FRAME_HEIGHT = 720
# ------------------------------------------------------------

# Состояние конвейера (эмуляция)
conveyor_running = True

def stop_conveyor_emulation():
    global conveyor_running
    if conveyor_running:
        conveyor_running = False
        print("[EMUL] Conveyor STOPPED (flag set).")
        # TODO: заменить эмуляцию реальной командой на ПЛК (через serial/ethernet/opcua)

# Загрузка модели
try:
    model = YOLO(MODEL_WEIGHTS)
    print(f"Loaded weights: {MODEL_WEIGHTS}")
except Exception as e:
    print("Failed to load custom weights, loading fallback model:", e)
    model = YOLO(FALLBACK_MODEL)

# Настройка камеры
cap = cv2.VideoCapture(VIDEO_SRC, cv2.CAP_DSHOW)  # CAP_DSHOW полезен на Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
time.sleep(0.5)  # нехитрая пауза для инициализации камеры

if not cap.isOpened():
    raise RuntimeError("Не удалось открыть камеру. Проверьте индекс и доступность устройства.")

# Tkinter GUI
root = tk.Tk()
root.title("Ore fraction monitor")

# Canvas для видео
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Не удалось получить кадр с камеры при старте")
h, w = frame.shape[:2]
canvas = tk.Canvas(root, width=w, height=h)
canvas.pack()

status_label = tk.Label(root, text="Состояние: OK", font=("Arial", 14))
status_label.pack()

# Для хранения PhotoImage чтобы не удалялся сборщиком
imgtk_ref = None

def process_frame_and_detect(frame_bgr):
    """
    Выполняет предсказание на кадре и возвращает список детекций:
    [{'class':name, 'conf':float, 'bbox':(x1,y1,x2,y2), 'size_mm':float}, ...]
    """
    # Ultralytics принимает BGR numpy arrays напрямую
    # predict возвращает Results, берем первый (batch size 1)
    res = model.predict(source=frame_bgr, conf=CONF_THRESHOLD, verbose=False, device='cpu', imgsz=640)
    results = res[0]  # один элемент
    detections = []
    if hasattr(results, 'boxes') and results.boxes is not None:
        for b in results.boxes:
            # b.xyxy: tensor [[x1,y1,x2,y2]], b.conf, b.cls
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            conf = float(b.conf[0].cpu().numpy()) if hasattr(b, 'conf') else float(b.conf)
            cls_id = int(b.cls[0].cpu().numpy()) if hasattr(b, 'cls') else int(b.cls)
            cls_name = model.names[cls_id] if model.names else str(cls_id)
            x1, y1, x2, y2 = xyxy.tolist()
            px_w = x2 - x1
            px_h = y2 - y1
            max_px = max(px_w, px_h)
            size_mm = max_px / PIXELS_PER_MM
            detections.append({
                'class': cls_name,
                'conf': conf,
                'bbox': (x1, y1, x2, y2),
                'size_mm': size_mm
            })
    return detections

def update_loop():
    global imgtk_ref, conveyor_running
    ret, frame = cap.read()
    if not ret:
        root.after(30, update_loop)
        return

    # детекция (можно вынести в отдельный thread, но тогда нужно безопасно передавать данные в GUI)
    detections = process_frame_and_detect(frame)

    alarm = False
    for d in detections:
        x1,y1,x2,y2 = d['bbox']
        label = f"{d['class']} {d['conf']:.2f} {d['size_mm']:.0f}mm"
        # цвет по умолчанию — зелёный
        color = (0,255,0)
        thickness = 2
        if d['class'] == 'foreign' or d['size_mm'] > SIZE_THRESHOLD_MM:
            alarm = True
            color = (0,0,255)
            thickness = 3
        # рисуем bbox и подпись (на RGB картинке)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if alarm and conveyor_running:
        stop_conveyor_emulation()

    # Отобразить статус
    if not conveyor_running:
        status_label.config(text="Состояние: КОНВЕЙЕР ОСТАНОВЛЕН", fg="red")
    elif alarm:
        status_label.config(text="Состояние: ТРЕВОГА", fg="orange")
    else:
        status_label.config(text="Состояние: OK", fg="green")

    # Конвертация BGR->RGB и отображение в Tkinter
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    imgtk_ref = imgtk  # сохранить референс
    canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    root.after(30, update_loop)

# Запуск цикла
root.after(0, update_loop)
root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), root.destroy()))
root.mainloop()
