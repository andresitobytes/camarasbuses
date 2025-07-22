from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt") 
results = model.train(data="coco-pose.yaml", epochs=100, imgsz=640)