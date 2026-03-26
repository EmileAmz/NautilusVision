from ultralytics import YOLO
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
WEIGHTS_DIR = r"C:\Users\eaime\OneDrive - USherbrooke\S7GRO\NautilusVision\datasets\Tests_Datasets_Roboflow\Data_1\runs\detect\train2\weights\best.pt"
IMAGE_DIR = r"C:\Users\eaime\OneDrive - USherbrooke\S7GRO\NautilusVision\datasets\Test_Piscine_a_annoter\Tests_march_18\rgb"

model = YOLO("yolov8n-obb.pt")  # or yolov8s-obb.pt for better accuracy

# Train
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="cpu",        # or "cpu"
    workers=4,
    name="obb_model"
)

