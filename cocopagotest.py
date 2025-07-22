from ultralytics import YOLO
import torch
import math

model = YOLO("yolo11n-pose.pt")

results = model("imagenes_prueba/imagen_pago_2.jpg")

keypoints = results[0].keypoints.data  # Tensor shape: [N, 17, 3]

# siempre dos personas detectadas
if keypoints.shape[0] >= 2:
    persona1 = keypoints[0]
    persona2 = keypoints[1]

    # Extraer puntos de las muñecas
    mano1 = persona1[9][:2]  # left_wrist pasajero
    mano2 = persona2[10][:2] # right_wrist conductor

    distancia = torch.norm(mano1 - mano2).item()

    print(f"Distancia entre manos: {distancia:.2f} px")

    # rango de contacto hardcodeado (ajustarlo según la imagen)
    if distancia < 200:
        print("Pago detectado (contacto de manos)")
    else:
        print("No hay contacto")
else:
    print("No se detectaron dos personas")
