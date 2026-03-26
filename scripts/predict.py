from ultralytics import YOLO
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
WEIGHTS_DIR = r"C:\Users\eaime\OneDrive - USherbrooke\S7GRO\NautilusVision\datasets\Tests_Datasets_Roboflow\Data_1\runs\detect\train2\weights\best.pt"
IMAGE_DIR = r"C:\Users\eaime\OneDrive - USherbrooke\S7GRO\NautilusVision\datasets\Test_Piscine_a_annoter\Tests_march_18\rgb"

model = YOLO(WEIGHTS_DIR)

model.predict(
    source=IMAGE_DIR,
    save=True,
    conf=0.25
)

