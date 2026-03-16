from ultralytics import YOLO

model = YOLO("Data/runs/detect/train2/weights/best.pt")

model.predict(
    source="Data/valid/images",
    save=True,
    conf=0.25
)

