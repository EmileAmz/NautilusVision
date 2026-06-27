import os
import cv2

# ====== CHANGE THESE ======
INPUT_IMAGE_DIR = "/home/nautilus/GithubVision/Datasets/Ready_to_train/Total/images_filtered"
OUTPUT_IMAGE_DIR = "/home/nautilus/GithubVision/Datasets/Ready_to_train/Total/images_filtered_resolution_75"

# Optional: copy labels as-is
INPUT_LABEL_DIR = "/home/nautilus/GithubVision/Datasets/Ready_to_train/Total/labels_filtered"
OUTPUT_LABEL_DIR = "/home/nautilus/GithubVision/Datasets/Ready_to_train/Total/labels_filtered_resolution_75"
# ==========================

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# ---- Resize images ----
for filename in os.listdir(INPUT_IMAGE_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(INPUT_IMAGE_DIR, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to read {filename}")
            continue

        h, w = img.shape[:2]
        scale = 0.75
        resized = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

        out_path = os.path.join(OUTPUT_IMAGE_DIR, filename)
        cv2.imwrite(out_path, resized)

# ---- Copy labels unchanged ----
for filename in os.listdir(INPUT_LABEL_DIR):
    if filename.endswith(".txt"):
        src = os.path.join(INPUT_LABEL_DIR, filename)
        dst = os.path.join(OUTPUT_LABEL_DIR, filename)

        with open(src, "r") as f_src, open(dst, "w") as f_dst:
            f_dst.write(f_src.read())

print("Done resizing images and copying labels.")