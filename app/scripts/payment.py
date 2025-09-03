from ultralytics import YOLO
import torch
import cv2
import os

def procesar_pago(video_input_path: str, video_output_path: str):
    model = YOLO("yolo11n-pose.pt")

    cap = cv2.VideoCapture(video_input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(video_output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    # Configuración
    FRAMES_CONFIRMACION = 5
    DISTANCIA_UMBRAL = width * 0.35

    # Variables de control
    conteo_pagos = 0
    tiempos_pago = []
    contador_frames_pagados = 0
    pagado_confirmado = False
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.6, verbose=False)
        annotated_frame = results[0].plot()
        keypoints = results[0].keypoints

        if keypoints is not None and keypoints.data.shape[0] >= 2:
            data = keypoints.data  # Tensor [N, 17, 3]
            persona1 = data[0]
            persona2 = data[1]

            mano1 = persona1[9][:2]  # left_wrist pasajero
            mano2 = persona2[10][:2] # right_wrist conductor

            distancia = torch.norm(mano1 - mano2).item()

            if distancia < DISTANCIA_UMBRAL:
                contador_frames_pagados += 1
            else:
                contador_frames_pagados = 0

            if contador_frames_pagados >= FRAMES_CONFIRMACION and not pagado_confirmado:
                pagado_confirmado = True
                conteo_pagos += 1
                segundos = round(frame_id / fps, 2)
                tiempos_pago.append(segundos)
        else:
            contador_frames_pagados = 0

        # Dibujar información en el frame
        texto = "PAGADO" if pagado_confirmado else "NO PAGADO"
        color = (0, 255, 0) if pagado_confirmado else (0, 0, 255)

        cv2.rectangle(annotated_frame, (20, 20), (300, 90), color, -1)
        cv2.putText(annotated_frame, texto, (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        out.write(annotated_frame)
        frame_id += 1

    cap.release()
    out.release()

    # Retornar resultados
    return conteo_pagos, tiempos_pago, video_output_path
