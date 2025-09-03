import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

VIDEO_PATH = 'videos_prueba/chofercorto.mp4'
OUTPUT_PATH = 'resultados_pose_video/ingresos_chofercorto.mp4'

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

# 🔹 Línea vertical de referencia (mitad de la puerta)
LINEA_X = int((PUERTA_X1 + PUERTA_X2) / 2)
TOLERANCIA = 10

# 🔹 Contadores
conteo_subidas = 0
conteo_bajadas = 0
tiempos_subidas = []
tiempos_bajadas = []

# 🔹 Historial por persona
lado_anterior = {}    # id -> 'izquierda' o 'derecha'
estado_cruce = {}     # id -> None / 'entrando' / 'saliendo'

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
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        person_id = int(box.id.item()) if box.id is not None else None

        if person_id is None or h < ALTURA_MINIMA:
            continue

        # Ignorar conductor
        if cx < ZONA_CONDUCTOR_X and cy > ZONA_CONDUCTOR_Y:
            continue

        # Revisar si está en la puerta
        en_puerta = PUERTA_X1 <= cx <= PUERTA_X2 and PUERTA_Y1 <= cy <= PUERTA_Y2
        if not en_puerta:
            continue

        # 🔹 Determinar lado respecto a la línea X
        if cx < LINEA_X - TOLERANCIA:
            lado_actual = "izquierda"
        elif cx > LINEA_X + TOLERANCIA:
            lado_actual = "derecha"
        else:
            lado_actual = lado_anterior.get(person_id, None)

        if person_id not in lado_anterior:
            lado_anterior[person_id] = lado_actual
            estado_cruce[person_id] = None
            continue

        # 🔹 Detectar cruce
        if lado_actual is not None and lado_actual != lado_anterior[person_id] and estado_cruce[person_id] is None:
            if lado_anterior[person_id] == "izquierda" and lado_actual == "derecha":
                conteo_subidas += 1
                tiempos_subidas.append(round(frame_id / fps, 2))
                estado_cruce[person_id] = "entrando"
            elif lado_anterior[person_id] == "derecha" and lado_actual == "izquierda":
                conteo_bajadas += 1
                tiempos_bajadas.append(round(frame_id / fps, 2))
                estado_cruce[person_id] = "saliendo"

            lado_anterior[person_id] = lado_actual

        # 🔹 Resetear estado
        if estado_cruce[person_id] == "entrando" and lado_actual == "derecha":
            estado_cruce[person_id] = None
        elif estado_cruce[person_id] == "saliendo" and lado_actual == "izquierda":
            estado_cruce[person_id] = None

        # 🔹 Dibujar bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{person_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # 🔹 Dibujar zona y línea
    cv2.rectangle(frame, (PUERTA_X1, PUERTA_Y1), (PUERTA_X2, PUERTA_Y2), (0, 255, 255), 2)
    cv2.line(frame, (LINEA_X, PUERTA_Y1), (LINEA_X, PUERTA_Y2), (0, 0, 255), 3)

    # 🔹 Mostrar conteos
    cv2.putText(frame, f"Entradas: {conteo_subidas}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"Salidas: {conteo_bajadas}", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    out.write(frame)
    cv2.imshow("Ingreso/Bajada", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Entradas totales:", conteo_subidas)
print("Tiempos de entradas (segundos):", tiempos_subidas)
print("Salidas totales:", conteo_bajadas)
print("Tiempos de salidas (segundos):", tiempos_bajadas)
