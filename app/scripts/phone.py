# scripts/actividad_telefono.py

import cv2
import os
import math
from ultralytics import YOLO

# Parámetros
FRAME_SKIP = 3
DISTANCE_THRESHOLD = 100
FPS = 30  # Puedes ajustar dinámicamente si quieres leerlo del video
CLASSES = {
    0: "person",
    67: "cell phone"
}

# Funciones auxiliares
def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def distance(box1, box2):
    c1 = center(box1)
    c2 = center(box2)
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

def procesar_video(video_path: str, output_path: str) -> dict:
    model = YOLO('yolov8x.pt')
    cap = cv2.VideoCapture(video_path)

    # Detectar FPS del video real
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    global FPS
    if real_fps > 0:
        FPS = int(real_fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))

    frame_idx = 0
    tiempos_uso = []
    intervalos_uso = []

    uso_detectado_anterior = False
    inicio_intervalo = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        results = model(frame)[0]
        people = []
        phones = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls_id == 0:
                people.append((x1, y1, x2, y2))
            elif cls_id == 67:
                phones.append((x1, y1, x2, y2))

        uso_detectado = False

        if people:
            person_box = max(people, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            px1, py1, px2, py2 = person_box
            cabeza_y = py1 + (py2 - py1) * 0.4
            pecho_y = py1 + (py2 - py1) * 0.7

            for phone_box in phones:
                phone_center = center(phone_box)
                if phone_center[1] < pecho_y:
                    d = distance(person_box, phone_box)
                    if d < DISTANCE_THRESHOLD:
                        uso_detectado = True
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                        cv2.rectangle(frame, (phone_box[0], phone_box[1]),
                                      (phone_box[2], phone_box[3]), (0, 0, 255), 2)
                        cv2.putText(frame, "USO DE TELEFONO", (px1, py1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if uso_detectado:
            segundo = round(frame_idx / FPS, 2)
            tiempos_uso.append(segundo)
            if not uso_detectado_anterior:
                inicio_intervalo = segundo
        else:
            if uso_detectado_anterior and inicio_intervalo is not None:
                fin_intervalo = round(frame_idx / FPS, 2)
                intervalos_uso.append((inicio_intervalo, fin_intervalo))
                inicio_intervalo = None

        uso_detectado_anterior = uso_detectado
        out.write(frame)
        frame_idx += 1

    # Cerrar último intervalo
    if uso_detectado_anterior and inicio_intervalo is not None:
        fin_intervalo = round(frame_idx / FPS, 2)
        intervalos_uso.append((inicio_intervalo, fin_intervalo))

    cap.release()
    out.release()

    tiempo_total = round(len(tiempos_uso) * FRAME_SKIP / FPS, 2)

    return {
        "output_path": output_path,
        "tiempo_uso_telefono": tiempo_total,
        "intervalos_uso_telefono": intervalos_uso
    }
