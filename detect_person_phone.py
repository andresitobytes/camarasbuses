import cv2
import os
import math
from ultralytics import YOLO

VIDEO_PATH = 'chofer_720p.mp4'
OUTPUT_DIR = 'frames_uso_telefono'
FRAME_SKIP = 3
DISTANCE_THRESHOLD = 100
FPS = 30  # Ajusta si tu video tiene otro framerate

CLASSES = {
    0: "person",
    67: "cell phone"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = YOLO('yolov8x.pt')

def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def distance(box1, box2):
    c1 = center(box1)
    c2 = center(box2)
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0

# ✅ Lista con tiempos de uso detectado
tiempos_uso = []

# ✅ Lista con información detallada de cada detección
detecciones = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % FRAME_SKIP != 0:
        frame_idx += 1
        continue

    results = model(frame)[0]
    people = []
    phones = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if cls_id == 0:
            people.append((x1, y1, x2, y2))
        elif cls_id == 67:
            phones.append((x1, y1, x2, y2))

    uso_detectado = False
    telefono_detectado = False
    distancia_px = None
    celular_arriba = False

    if people:
        person_box = max(people, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        px1, py1, px2, py2 = person_box
        cabeza_y = py1 + (py2 - py1) * 0.4
        pecho_y = py1 + (py2 - py1) * 0.7
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)

        for phone_box in phones:
            telefono_detectado = True
            cx1, cy1, cx2, cy2 = phone_box
            phone_center = center(phone_box)
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 0, 100), 1)

            if phone_center[1] < pecho_y:
                celular_arriba = True
                d = distance(person_box, phone_box)
                distancia_px = int(d)
                if d < DISTANCE_THRESHOLD:
                    uso_detectado = True
                    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
                    cv2.putText(frame, "USO DE TELEFONO", (px1, py1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if uso_detectado:
        segundo = round(frame_idx / FPS, 2)
        tiempos_uso.append(segundo)
        detecciones.append({
            "segundo": segundo,
            "telefono_detectado": telefono_detectado,
            "distancia_px": distancia_px,
            "celular_arriba": celular_arriba,
        })
        save_name = f"uso_frame_{frame_idx:05}.jpg"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        cv2.imwrite(save_path, frame)

    frame_idx += 1
    cv2.imshow("Detección", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ✅ Mostrar resultados
if tiempos_uso:
    tiempo_total = round(len(tiempos_uso) * FRAME_SKIP / FPS, 2)
    print(f"\n📱 Tiempo total de uso de teléfono: {tiempo_total} segundos")

    # ✅ Identificar intervalos continuos
    intervalos_uso = []
    inicio = tiempos_uso[0]
    for i in range(1, len(tiempos_uso)):
        if tiempos_uso[i] - tiempos_uso[i - 1] > FRAME_SKIP / FPS:
            fin = tiempos_uso[i - 1]
            intervalos_uso.append((inicio, fin))
            inicio = tiempos_uso[i]
    intervalos_uso.append((inicio, tiempos_uso[-1]))

    print("\n📊 Intervalos de uso detectados:")
    for i, (ini, fin) in enumerate(intervalos_uso):
        print(f"  {i+1}. Desde {ini:.2f}s hasta {fin:.2f}s (Duración: {round(fin - ini, 2)}s)")

else:
    print("📵 No se detectó uso de teléfono en ningún momento.")
