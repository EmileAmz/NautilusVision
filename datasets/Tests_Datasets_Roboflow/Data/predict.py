from ultralytics import YOLO

model = YOLO("DataTest/runs/detect/train2/weights/best.pt")

model.predict(
    source="Data/test/images",
    save=True,
    conf=0.25
)

