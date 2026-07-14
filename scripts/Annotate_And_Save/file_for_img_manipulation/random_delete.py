#!/usr/bin/env python3

import random
from pathlib import Path

# =====================================================
# CONFIGURATION
# =====================================================

FOLDER = Path(r"C:\Users\Xavier Lefebvre\Documents\dataset\13_juillet")  # <-- Modifier ce chemin

KEEP_RATIO = 0.95

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}

# =====================================================
# MAIN
# =====================================================

images = [
    f for f in FOLDER.iterdir()
    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
]

total = len(images)

if total == 0:
    print("Aucune image trouvée.")
    exit()

random.shuffle(images)

num_keep = int(total * KEEP_RATIO)
keep_set = set(images[:num_keep])

deleted = 0

for img in images:
    if img not in keep_set:
        img.unlink()
        deleted += 1

print(f"Images trouvées : {total}")
print(f"Images conservées : {num_keep}")
print(f"Images supprimées : {deleted}")
print("Terminé.")