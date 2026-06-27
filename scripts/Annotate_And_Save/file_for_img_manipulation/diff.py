#!/usr/bin/env python3

from pathlib import Path

# ===== CONFIG =====
txt_folder = Path("/home/nautilus/GithubVision/Datasets/Ready_to_train/Total/labels_filtered")
image_folder = Path("/home/nautilus/GithubVision/Datasets/Ready_to_train/Total/images_filtered")

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ==================

# Noms de base des fichiers txt
txt_names = {
    f.stem
    for f in txt_folder.iterdir()
    if f.is_file() and f.suffix.lower() == ".txt"
}

# Noms de base des images
image_names = {
    f.stem
    for f in image_folder.iterdir()
    if f.is_file() and f.suffix.lower() in image_extensions
}

missing_images = sorted(txt_names - image_names)
missing_txt = sorted(image_names - txt_names)

print("=" * 60)
print(f"Labels (.txt)      : {len(txt_names)}")
print(f"Images             : {len(image_names)}")
print("=" * 60)

print(f"\nTXT sans image ({len(missing_images)})")
for name in missing_images:
    print(name)

print(f"\nImages sans TXT ({len(missing_txt)})")
for name in missing_txt:
    print(name)