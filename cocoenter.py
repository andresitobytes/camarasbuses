from ultralytics import YOLO
import torch
import cv2
import os
from collections import defaultdict

model = YOLO("yolo11n-pose.pt")

ruta_video = "videos_prueba/video_pago.mov"
video_output = "resultados_pose_video/output_ingresos.mp4"

cap = cv2.VideoCapture(ruta_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
os.makedirs("resultados_pose_video", exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

# linea que detecta la entrada de los pasajeros (nota: no está bien calibrado, hay que buscar una forma de estandarizar esto)
linea_entrada_x = int(width * 0.85)

trayectorias = defaultdict(list)
personas_ingresadas = set()

frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=0.6, verbose=False)
    annotated_frame = results[0].plot()

    boxes = results[0].boxes
    if boxes.id is not None:
        for i in range(len(boxes.id)):
            id_persona = int(boxes.id[i].item())
            clase = int(boxes.cls[i].item())
            if clase != 0:
                continue  # Solo personas

            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            trayectorias[id_persona].append((cx, cy))

            # detectar cruce de derecha a izquierda (entrada al bus)
            trayectoria = trayectorias[id_persona]
            if len(trayectoria) >= 2 and id_persona not in personas_ingresadas:
                x_ant = trayectoria[-2][0]
                x_act = trayectoria[-1][0]

                if x_ant > linea_entrada_x and x_act <= linea_entrada_x:
                    personas_ingresadas.add(id_persona)
                    print(f"Persona ID {id_persona} ingresó al bus")

    # dibujar línea vertical de entrada
    cv2.line(annotated_frame, (linea_entrada_x, 0), (linea_entrada_x, height), (0, 255, 255), 2)
    
    # Mostrar número de personas ingresadas
    texto = f"Ingresos: {len(personas_ingresadas)}"
    cv2.rectangle(annotated_frame, (20, 20), (280, 90), (255, 140, 0), -1)
    cv2.putText(annotated_frame, texto, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    out.write(annotated_frame)
    frame_id += 1
# NO FUNCIONA BIEN, EL ANGULO DE LAS CAMARAS LO HACE DIFICIL DE TRACKEAR
cap.release()
out.release()
print("Video procesado y guardado en:", video_output)
