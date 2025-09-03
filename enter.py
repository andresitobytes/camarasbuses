import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

VIDEO_PATH = 'videos_prueba/chofercorto.mp4'
OUTPUT_PATH = 'resultados_pose_video/ingresos_chofercorto.mp4'
FPS = 60
INGRESO_X = 0.75  # Línea virtual (en porcentaje del ancho)

cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

linea_x = int(width * INGRESO_X)

# 🔹 Contadores
conteo_subidas = 0
conteo_bajadas = 0
tiempos_subidas = []
tiempos_bajadas = []
ids_contados_subida = set()
ids_contados_bajada = set()

historial_cx = {}  # id -> último cx

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
        w, h = x2 - x1, y2 - y1
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        person_id = int(box.id.item()) if box.id is not None else None

        if person_id is None or h < ALTURA_MINIMA:
            continue

        # Ignorar al conductor
        if cx < ZONA_CONDUCTOR_X and cy > ZONA_CONDUCTOR_Y:
            continue  

        cx_prev = historial_cx.get(person_id, None)
        historial_cx[person_id] = cx  # Actualizar histórico

        if cx_prev is not None:
            # 🚍 Subida: cruza de izquierda a derecha
            if (
                cx_prev < linea_x <= cx
                and (cx - cx_prev) > 5
                and person_id not in ids_contados_subida
            ):
                conteo_subidas += 1
                ids_contados_subida.add(person_id)
                segundos = round(frame_id / FPS, 2)
                tiempos_subidas.append(segundos)

            # 🚶‍♂️ Bajada: cruza de derecha a izquierda
            elif (
                cx_prev > linea_x >= cx
                and (cx_prev - cx) > 5
                and person_id not in ids_contados_bajada
            ):
                conteo_bajadas += 1
                ids_contados_bajada.add(person_id)
                segundos = round(frame_id / FPS, 2)
                tiempos_bajadas.append(segundos)

        # Dibujar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Dibujar línea de referencia
    cv2.line(frame, (linea_x, 0), (linea_x, height), (0, 255, 255), 2)

    # Mostrar conteos en pantalla
    cv2.putText(frame, f"Subidas: {conteo_subidas}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"Bajadas: {conteo_bajadas}", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    out.write(frame)
    cv2.imshow("Ingreso/Bajada", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# ✅ Resultados finales
print("Subidas totales:", conteo_subidas)
print("Tiempos de subidas (segundos):", tiempos_subidas)
print("Bajadas totales:", conteo_bajadas)
print("Tiempos de bajadas (segundos):", tiempos_bajadas)
