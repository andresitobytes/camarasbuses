from ultralytics import YOLO
import torch
import cv2
import os

model = YOLO("yolo11n-pose.pt")

ruta_video = "videos_prueba/video_pago.mov"
video_output = "resultados_pose_video/output_pago.mp4"
cap = cv2.VideoCapture(ruta_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
os.makedirs("resultados_pose_video", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))
pagado = False
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame,conf=0.6, verbose=False)

    annotated_frame = results[0].plot()
    keypoints = results[0].keypoints

    if keypoints is not None and keypoints.data.shape[0] >= 2:
        data = keypoints.data  # Tensor [N, 17, 3]
        persona1 = data[0]
        persona2 = data[1]

        mano1 = persona1[9][:2]  # left_wrist pasajero
        mano2 = persona2[10][:2] # right_wrist conductor

        distancia = torch.norm(mano1 - mano2).item()

        if distancia < width*0.35: # proporción para detectar proximidad entre muñecas (así se ajusta a calidades de video), ajustar si es necesario!!
            print(f"Frame {frame_id}: Pago detectado (distancia: {distancia:.1f}px)")
            pagado = True
        else:
            print(f"Frame {frame_id}: No hay contacto (distancia: {distancia:.1f}px)")
    if pagado:
        texto = "PAGADO"
        color = (0, 255, 0)
    else:
        texto = "NO PAGADO"
        color = (0, 0, 255)
    cv2.rectangle(annotated_frame, (20, 20), (300, 90), color, -1)
    cv2.putText(annotated_frame, texto, (30, 70),
    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    out.write(annotated_frame)
    frame_id += 1
# MEJORA A HACER: HACER QUE EL FRAME DE PAGADO SEA UNA CANTIDAD X DE VECES SEGUIDAS PARA ASI ASEGURARNOS QUE ES UN PAGO Y NO ES UNA IMPRESICIÓN DEL MODELO
cap.release()
out.release()
print("Video procesado y guardado en:", video_output)
