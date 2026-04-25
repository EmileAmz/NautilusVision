import shutil
from pathlib import Path

# ===================== CONFIG =====================
DATASET1_IMAGES = Path("C:/Users/eaime/Documents/S7GRO/PHOTO_14AVRIL/images")
DATASET1_LABELS = Path("C:/Users/eaime/Documents/S7GRO/PHOTO_14AVRIL/labels_obb")

DATASET2_IMAGES = Path("C:/Users/eaime/Documents/S7GRO/Nautilus_images_filtered_and_regular/images")
DATASET2_LABELS = Path("C:/Users/eaime/Documents/S7GRO/Nautilus_images_filtered_and_regular/labels_obb")

OUTPUT_IMAGES = Path("C:/Users/eaime/Documents/S7GRO/Merged_dataset/Total/images")
OUTPUT_LABELS = Path("C:/Users/eaime/Documents/S7GRO/Merged_dataset/Total/labels_obb")

PREFIX_DS2 = "ds2"  # used ONLY if collision happens
# ==================================================


def copy_dataset1(images_dir, labels_dir, out_images, out_labels):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in valid_exts]

    print(f"\nCopying dataset 1: {len(image_paths)} images")

    for img_path in image_paths:
        # Copy image as-is
        out_img = out_images / img_path.name
        shutil.copy(img_path, out_img)

        # Copy label
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy(label_path, out_labels / label_path.name)
        else:
            print(f"  Warning: missing label for {img_path.name}")


def copy_dataset2(images_dir, labels_dir, out_images, out_labels, prefix):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in valid_exts]

    print(f"\nCopying dataset 2: {len(image_paths)} images")

    for img_path in image_paths:
        target_img = out_images / img_path.name

        # 🔥 Check collision
        if target_img.exists():
            new_name = f"{prefix}_{img_path.name}"
            print(f"  Collision: {img_path.name} → {new_name}")
        else:
            new_name = img_path.name

        out_img = out_images / new_name
        shutil.copy(img_path, out_img)

        # Handle label
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            #if target_img.exists():
            #    new_label_name = f"{prefix}_{label_path.name}"
            #else:
            new_label_name = label_path.name

            out_label = out_labels / new_label_name
            shutil.copy(label_path, out_label)
        else:
            print(f"  Warning: missing label for {img_path.name}")


def main():
    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

    # Step 1: copy dataset 1 normally
    copy_dataset1(DATASET1_IMAGES, DATASET1_LABELS, OUTPUT_IMAGES, OUTPUT_LABELS)

    # Step 2: copy dataset 2 with collision handling
    copy_dataset2(DATASET2_IMAGES, DATASET2_LABELS, OUTPUT_IMAGES, OUTPUT_LABELS, PREFIX_DS2)

    print("\nMerge complete.")


if __name__ == "__main__":
    main()