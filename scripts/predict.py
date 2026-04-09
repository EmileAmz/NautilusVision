from ultralytics import YOLO
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
IMAGE_DIR = Path("C:/Users/eaime/Documents/S7GRO/Nautilus images sim split/test/images")
LABEL_DIR = SCRIPT_DIR / "datasets/Test_Piscine_split/Tests_march_18_bbox/test/labels"
WEIGHTS_DIR = SCRIPT_DIR / "runs/detect/bbox_sim/weights/best.pt"

model = YOLO(WEIGHTS_DIR)

model.predict(
    source=IMAGE_DIR,
    save=True,
    conf=0.25
)

