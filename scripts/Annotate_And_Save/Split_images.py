import os
import random
import shutil
from pathlib import Path

# -------- CONFIG --------
IMAGE_DIR = Path("C:/Users/eaime/Documents/S7GRO/Nautilus images sim split/320p/Total/images")
LABEL_DIR = Path("C:/Users/eaime/Documents/S7GRO/Nautilus images sim split/320p/Total/labels")
DATA_YAML = Path("C:/Users/eaime/Documents/S7GRO/Nautilus images sim split/320p/Total/data.yaml")

OUTPUT_DIR = Path("C:/Users/eaime/Documents/S7GRO/Nautilus images sim split/320p")

SPLIT_RATIO = (0.7, 0.15, 0.15)  # train, val, test
SEED = 42

def normalize_stem(name: str):
    return name.replace("ds2_", "")

# -------- SETUP --------
random.seed(SEED)

image_files = list(IMAGE_DIR.glob("*.*"))
image_files = [f for f in image_files if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

image_stems = {f.stem for f in IMAGE_DIR.glob("*.*")}
label_stems = {f.stem for f in LABEL_DIR.glob("*.txt")}

random.shuffle(image_files)

n = len(image_files)
n_train = int(n * SPLIT_RATIO[0])
n_val = int(n * SPLIT_RATIO[1])

train_files = image_files[:n_train]
val_files = image_files[n_train:n_train + n_val]
test_files = image_files[n_train + n_val:]

splits = {
    "train": train_files,
    "val": val_files,
    "test": test_files
}

# -------- CREATE FOLDERS --------
for split in splits.keys():
    (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

# -------- COPY FILES --------
for split, files in splits.items():
    for img_path in files:
        label_path = LABEL_DIR / (normalize_stem(img_path.stem) + ".txt")

        # Copy image
        shutil.copy(img_path, OUTPUT_DIR / split / "images" / img_path.name)

        # Copy label if exists
        if label_path.exists():
            shutil.copy(label_path, OUTPUT_DIR / split / "labels" / label_path.name)
        else:
            print(f"⚠️ Warning: No label for {img_path.name}")


if DATA_YAML.exists():
    shutil.copy(DATA_YAML, OUTPUT_DIR / "data.yaml")
    print("✅ data.yaml copied to dataset root")
else:
    print("⚠️ data.yaml not found, skipping")

print("✅ Dataset successfully split!")