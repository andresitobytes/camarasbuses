from ultralytics import YOLO
import os

model = YOLO("yolo11n-pose.pt")  # Modelo de pose estimation

carpeta_videos = "videos_prueba"

carpeta_salida = "resultados_pose_video"
os.makedirs(carpeta_salida, exist_ok=True)

for nombre_archivo in os.listdir(carpeta_videos):
    if nombre_archivo.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        ruta_video = os.path.join(carpeta_videos, nombre_archivo)
        print(f"Procesando video: {ruta_video}")

        results = model.predict(
            source=ruta_video,
            save=True,
            save_dir=carpeta_salida,
            conf=0.3,
            stream=False
        )

print("Procesamiento de video completado. Resultados guardados en:", carpeta_salida)
