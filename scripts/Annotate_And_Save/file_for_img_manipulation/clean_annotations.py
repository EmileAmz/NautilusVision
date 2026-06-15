import os

# ======== CHANGE THESE ========
images_dir = r"C:\Users\eaime\Documents\S7GRO\22-04-26\rgb-oakd"
labels_dir = r"C:\Users\eaime\Documents\S7GRO\22-04-26\labels_obb"
# =============================

# Supported image extensions
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def get_base_names(folder, extensions=None):
    files = os.listdir(folder)
    base_names = set()

    for f in files:
        name, ext = os.path.splitext(f)
        if extensions:
            if ext.lower() in extensions:
                base_names.add(name)
        else:
            base_names.add(name)

    return base_names


# Get base filenames (without extension)
image_bases = get_base_names(images_dir, image_extensions)
label_bases = get_base_names(labels_dir)  # assuming labels are .txt

# Find mismatches
labels_without_images = label_bases - image_bases
images_without_labels = image_bases - label_bases

# Delete labels without images
for base in labels_without_images:
    label_path = os.path.join(labels_dir, base + ".txt")
    if os.path.exists(label_path):
        os.remove(label_path)
        print(f"Deleted label: {label_path}")

# Delete images without labels
for base in images_without_labels:
    for ext in image_extensions:
        image_path = os.path.join(images_dir, base + ext)
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted image: {image_path}")

# Summary
print("\n=== Summary ===")
print(f"Removed {len(labels_without_images)} orphan labels")
print(f"Removed {len(images_without_labels)} orphan images")
print("Dataset is now aligned ✅")