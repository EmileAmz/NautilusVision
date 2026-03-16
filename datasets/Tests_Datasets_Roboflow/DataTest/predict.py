from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")

model.predict(
    source="valid/images",
    save=True,
    conf=0.25
)

