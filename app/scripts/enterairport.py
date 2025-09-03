import cv2
from ultralytics import YOLO
import torch
import tempfile

def procesar_video(video_path: str, output_path: str = None):
    ingreso_x = 0.95
    skip_frames = 1
    scale = 1.0
    verbose = False

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_path = tmp.name
        tmp.close()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt").to(device)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    width_s = int(width * scale)
    height_s = int(height * scale)

    out = cv2.VideoWriter(output_path, fourcc, fps, (width_s, height_s))

    linea_x = int(width_s * ingreso_x)

    conteo = 0
    ids_contados = set()
    historial_cx = {}

    ZONA_CONDUCTOR_X = 0.3 * width_s
    ZONA_CONDUCTOR_Y = 0.6 * height_s
    ALTURA_MINIMA = 0.15 * height_s

    tiempos_ingreso = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if frame_id % skip_frames != 0:
            continue

        if scale != 1.0:
            frame = cv2.resize(frame, (width_s, height_s))

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

            if cx < ZONA_CONDUCTOR_X and cy > ZONA_CONDUCTOR_Y:
                continue  # Ignorar conductor

            cx_prev = historial_cx.get(person_id, None)
            historial_cx[person_id] = cx

            if (
                cx_prev is not None
                and cx_prev < linea_x <= cx
                and cx > cx_prev
                and (cx - cx_prev) > 5
                and person_id not in ids_contados
            ):
                conteo += 1
                ids_contados.add(person_id)
                segundos = round(frame_id / fps, 2)
                tiempos_ingreso.append(segundos)
                if verbose:
                    print(f"✅ Persona #{conteo} (ID {person_id}) ingresó en el segundo {segundos} (frame {frame_id})")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.line(frame, (linea_x, 0), (linea_x, height_s), (0, 255, 255), 2)
        cv2.putText(frame, f"Ingresos: {conteo}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        out.write(frame)

    cap.release()
    out.release()

    return {
        "ingresos": conteo,
        "tiempos_ingreso": tiempos_ingreso,
        "fps_video": fps,
        "frames_procesados": frame_id // skip_frames,
        "output_path": output_path
    }
