# scripts/puerta_abierta.py

import cv2
import numpy as np
from ultralytics import YOLO

# Define ROI y zona del conductor
ROI = (740, 80, 900, 350)
DRIVER_AREA = (740, 80, 780, 350)

# Umbrales
MOVEMENT_THRESHOLD = 1.5
CHANGE_THRESHOLD = 40

def box_intersect(boxA, boxB):
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB
    return not (xA2 < xB1 or xB2 < xA1 or yA2 < yB1 or yB2 < yA1)

def person_in_roi(person_boxes, roi, exclude_zone=None):
    rx1, ry1, rx2, ry2 = roi
    for (x1, y1, x2, y2) in person_boxes:
        if exclude_zone and box_intersect((x1, y1, x2, y2), exclude_zone):
            continue
        if x2 >= rx1 and x1 <= rx2 and y2 >= ry1 and y1 <= ry2:
            return True
    return False

def procesar_video(video_path: str, output_path: str) -> dict:
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("No se pudo leer el video.")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_roi = prev_gray[ROI[1]:ROI[3], ROI[0]:ROI[2]]

    # Datos a recolectar
    frames_abierta = []
    intervalos_puerta_abierta = []
    frame_idx = 1
    puerta_abierta_anterior = False
    inicio_intervalo = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[ROI[1]:ROI[3], ROI[0]:ROI[2]]

        flow = cv2.calcOpticalFlowFarneback(prev_roi, roi, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_motion = np.mean(mag)

        diff = cv2.absdiff(prev_roi, roi)
        mean_diff = np.mean(diff)

        results = model(frame)
        person_boxes = []
        for r in results:
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                if int(cls) == 0:  # persona
                    x1, y1, x2, y2 = map(int, box)
                    person_boxes.append((x1, y1, x2, y2))

        puerta_abierta = False
        if mean_motion > MOVEMENT_THRESHOLD and not person_in_roi(person_boxes, ROI, exclude_zone=DRIVER_AREA):
            puerta_abierta = True
            estado = "PUERTA ABIERTA EN MOVIMIENTO"
            cv2.putText(frame, estado, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        elif mean_diff > CHANGE_THRESHOLD:
            estado = "Puerta cambió de estado"
        else:
            estado = "Puerta estable"

        # Registrar apertura
        if puerta_abierta:
            segundo = round(frame_idx / fps, 2)
            frames_abierta.append(segundo)
            if not puerta_abierta_anterior:
                inicio_intervalo = segundo
        else:
            if puerta_abierta_anterior and inicio_intervalo is not None:
                fin_intervalo = round(frame_idx / fps, 2)
                intervalos_puerta_abierta.append((inicio_intervalo, fin_intervalo))
                inicio_intervalo = None

        puerta_abierta_anterior = puerta_abierta

        # Dibuja ROI
        cv2.rectangle(frame, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (0, 255, 0), 2)
        out.write(frame)
        prev_roi = roi.copy()
        frame_idx += 1

    if puerta_abierta_anterior and inicio_intervalo is not None:
        fin_intervalo = round(frame_idx / fps, 2)
        intervalos_puerta_abierta.append((inicio_intervalo, fin_intervalo))

    cap.release()
    out.release()

    tiempo_total_puerta_abierta = round(len(frames_abierta) / fps, 2)

    return {
        "output_path": output_path,
        "tiempo_puerta_abierta": tiempo_total_puerta_abierta,
        "intervalos_puerta": intervalos_puerta_abierta
    }
