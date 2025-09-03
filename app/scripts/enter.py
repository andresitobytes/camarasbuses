import cv2
from ultralytics import YOLO

def procesar_ingresos_rapido(video_path: str, skip_frames=2, scale=0.5):
    model = YOLO("yolov8n.pt").to("cuda:0")  # GPU si disponible

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    width_s = int(width * scale)
    height_s = int(height * scale)
    linea_x = int(width_s * 0.79)

    conteo_total = 0
    tiempos_ingreso = []
    ids_contados = set()
    historial_cx = {}

    ZONA_CONDUCTOR_X = int(0.3 * width_s)
    ZONA_CONDUCTOR_Y = int(0.6 * height_s)
    ALTURA_MINIMA = int(0.15 * height_s)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_id % skip_frames != 0:
            continue

        frame = cv2.resize(frame, (width_s, height_s))

        results = model.track(frame, tracker="botsort.yaml")[0]
        if results is None or not hasattr(results, "boxes"):
            continue

        for box in results.boxes:
            if int(box.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            person_id = int(box.id.item()) if box.id is not None else None

            if person_id is None or h < ALTURA_MINIMA:
                continue

            if cx < ZONA_CONDUCTOR_X and cy > ZONA_CONDUCTOR_Y:
                continue

            cx_prev = historial_cx.get(person_id)
            historial_cx[person_id] = cx

            if (
                cx_prev is not None
                and cx_prev < linea_x <= cx
                and cx > cx_prev
                and (cx - cx_prev) > 5
                and person_id not in ids_contados
            ):
                conteo_total += 1
                ids_contados.add(person_id)
                segundos = round(frame_id / fps, 2)
                tiempos_ingreso.append(segundos)

    cap.release()

    return {
        "ingresos": conteo_total,
        "tiempos_ingreso": tiempos_ingreso
    }
