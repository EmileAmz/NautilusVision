from ultralytics import YOLO
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
WEIGHTS_DIR = SCRIPT_DIR / "datasets/Tests_Datasets_Roboflow/Data_1/runs/detec/train2/weights/best.pt"

model = YOLO(WEIGHTS_DIR)

model.predict(
    source="valid/images",
    save=True,
    conf=0.25
)

