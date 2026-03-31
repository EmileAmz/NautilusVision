from ultralytics import YOLO
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
# IMAGE_DIR = SCRIPT_DIR / "datasets/Test_Piscine_split/Tests_march_18_bbox/images"
# LABEL_DIR = SCRIPT_DIR / "datasets/Test_Piscine_split/Tests_march_18_bbox/labels"
# DATA_YAML = SCRIPT_DIR / "datasets/Test_Piscine_split/Tests_march_18_bbox/data.yaml"


IMAGE_DIR = Path("C:/Users/eaime/Documents/S7GRO/Nautilus images sim split")
LABEL_DIR = Path("C:/Users/eaime/Documents/S7GRO/Nautilus images sim split")
DATA_YAML = Path("C:/Users/eaime/Documents/S7GRO/Nautilus images sim split/data.yaml")

model = YOLO("yolov8n.pt")  # or yolov8s-obb.pt for better accuracy

# Train
model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    batch=16,
    device="cpu",        # or "cpu"
    workers=4,
    name="bbox_sim"
)

