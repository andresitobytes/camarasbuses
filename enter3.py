import cv2
from ultralytics import YOLO
import mediapipe as mp

# 🔹 Inicializar YOLO y MediaPipe Pose
model = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

VIDEO_PATH = 'videos_prueba/chofercorto.mp4'
OUTPUT_PATH = 'resultados_pose_video/ingresos_chofercorto_pose.mp4'

cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# 🔹 Zona de puerta
PUERTA_X1 = int(width * 0.70)
PUERTA_X2 = int(width * 0.90)
PUERTA_Y1 = int(height * 0.32)
PUERTA_Y2 = int(height * 0.82)

# 🔹 Contadores y buffers
conteo_subidas = 0
conteo_bajadas = 0
tiempos_subidas = []
tiempos_bajadas = []

# 🔹 Historial por persona
ultimo_centro_y = {}  # person_id -> última posición vertical (cy)
estado_cruce = {}     # person_id -> 'subiendo', 'bajando', None

# 🔹 Filtros
ZONA_CONDUCTOR_X = 0.3 * width
ZONA_CONDUCTOR_Y = 0.6 * height
ALTURA_MINIMA = 0.15 * height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    results = model.track(frame, persist=True, tracker="botsort.yaml")[0]

    if results is None or not hasattr(results, "boxes"):
        out.write(frame)
        continue

    for box in results.boxes:
        if int(box.cls[0]) != 0:
            continue  # Solo personas

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        h = y2 - y1
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        person_id = int(box.id.item()) if box.id is not None else None

        if person_id is None or h < ALTURA_MINIMA:
            continue

        # Ignorar conductor
        if cx < ZONA_CONDUCTOR_X and cy > ZONA_CONDUCTOR_Y:
            continue

        # Revisar si está en la puerta
        if not (PUERTA_X1 <= cx <= PUERTA_X2 and PUERTA_Y1 <= cy <= PUERTA_Y2):
            continue

        # 🔹 Recortar persona y pasar a MediaPipe Pose
        person_crop = frame[y1:y2, x1:x2]
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        pose_results = pose_detector.process(rgb_crop)

        if pose_results.pose_landmarks:
            # Extraer keypoints de pies y cadera
            lm = pose_results.pose_landmarks.landmark
            pie_derecho_y = lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h + y1
            pie_izquierdo_y = lm[mp_pose.PoseLandmark.LEFT_ANKLE].y * h + y1
            cadera_y = (lm[mp_pose.PoseLandmark.RIGHT_HIP].y + lm[mp_pose.PoseLandmark.LEFT_HIP].y)/2 * h + y1

            # Promedio de pies y cadera
            promedio_y = (pie_derecho_y + pie_izquierdo_y + cadera_y)/3

            # 🔹 Comparar con último centro
            if person_id in ultimo_centro_y:
                diff = promedio_y - ultimo_centro_y[person_id]
                if diff < -5 and estado_cruce.get(person_id) != 'subiendo':
                    conteo_subidas += 1
                    tiempos_subidas.append(round(frame_id / fps, 2))
                    estado_cruce[person_id] = 'subiendo'
                elif diff > 5 and estado_cruce.get(person_id) != 'bajando':
                    conteo_bajadas += 1
                    tiempos_bajadas.append(round(frame_id / fps, 2))
                    estado_cruce[person_id] = 'bajando'
                elif abs(diff) < 3:
                    estado_cruce[person_id] = None

            ultimo_centro_y[person_id] = promedio_y

        # 🔹 Dibujar bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{person_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # 🔹 Dibujar zona
    cv2.rectangle(frame, (PUERTA_X1, PUERTA_Y1), (PUERTA_X2, PUERTA_Y2), (0, 255, 255), 2)

    # 🔹 Mostrar conteos
    cv2.putText(frame, f"Subidas: {conteo_subidas}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"Bajadas: {conteo_bajadas}", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    out.write(frame)
    cv2.imshow("Ingreso/Bajada", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Subidas totales:", conteo_subidas)
print("Tiempos de subidas (segundos):", tiempos_subidas)
print("Bajadas totales:", conteo_bajadas)
print("Tiempos de bajadas (segundos):", tiempos_bajadas)
