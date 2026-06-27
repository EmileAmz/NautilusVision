from ultralytics import YOLO
from pathlib import Path

# SCRIPT_DIR = Path(__file__).parent.parent.resolve()
# IMAGE_DIR = SCRIPT_DIR / "datasets/Test_Piscine_split/Tests_march_18_bbox/images"
# LABEL_DIR = SCRIPT_DIR / "datasets/Test_Piscine_split/Tests_march_18_bbox/labels"
# DATA_YAML = SCRIPT_DIR / "datasets/Test_Piscine_split/Tests_march_18_bbox/data.yaml"


# IMAGE_DIR = Path("C:/Users/eaime/Documents/S7GRO/Nautilus images sim split/320p/")
# LABEL_DIR = Path("C:/Users/eaime/Documents/S7GRO/Merged_dataset/labels_obb")
DATA_YAML = Path(r"/home/nautilus/GithubVision/Datasets/Ready_to_train/Split/data.yaml")

model = YOLO("yolov8n.pt")  # or yolov8s-obb.pt for better accuracy

# Train
model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=960,
    batch=16,
    device=0,        # or "cpu"
    workers=0,
    name="bbox_25_juin_competition"
)

