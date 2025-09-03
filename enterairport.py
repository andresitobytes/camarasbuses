import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

VIDEO_PATH = 'videos_prueba/airport.mov'
OUTPUT_PATH = 'resultados_pose_video/ingresos_trackairport.mov'
FPS = 60
INGRESO_X = 0.95  # Línea virtual (en porcentaje del ancho)

cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

linea_x = int(width * INGRESO_X)

conteo = 0
ids_contados = set()
historial_cx = {}  # id -> último cx

# Parámetros para ignorar al conductor y detección mínima
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

        if cx < ZONA_CONDUCTOR_X and cy > ZONA_CONDUCTOR_Y:
            continue  # Ignorar conductor

        cx_prev = historial_cx.get(person_id, None)
        historial_cx[person_id] = cx  # Actualizar histórico

        # Condición estricta: cruce de izquierda a derecha + movimiento real hacia la derecha
        if (
            cx_prev is not None
            and cx_prev < linea_x <= cx
            and cx > cx_prev
            and (cx - cx_prev) > 5  # Evita conteo por pequeños movimientos
            and person_id not in ids_contados
        ):
            conteo += 1
            ids_contados.add(person_id)
            segundos = round(frame_id / FPS, 2)
            print(f"✅ Persona #{conteo} (ID {person_id}) ingresó en el segundo {segundos} (frame {frame_id})")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Dibujar línea y contador
    cv2.line(frame, (linea_x, 0), (linea_x, height), (0, 255, 255), 2)
    cv2.putText(frame, f"Ingresos: {conteo}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    out.write(frame)
    cv2.imshow("Ingreso", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
