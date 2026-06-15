import cv2
import numpy as np
import math
from PIL import Image
from pathlib import Path
import shutil

# ===================== CONFIG =====================
IMAGES_INPUT_DIR = Path("C:/Users/eaime/Documents/S7GRO/22-04-26/rgb-oakd")      # original images
LABELS_INPUT_DIR = Path("C:/Users/eaime/Documents/S7GRO/22-04-26/labels_obb")      # original labels

IMAGES_OUTPUT_DIR = Path("C:/Users/eaime/Documents/S7GRO/22-04-26/images")   # output images
LABELS_OUTPUT_DIR = Path("C:/Users/eaime/Documents/S7GRO/22-04-26/labels_obb_filtered")   # output labels

# ==================================================

THRESHOLD_RATIO = 2000
MIN_AVG_RED = 60
MAX_HUE_SHIFT = 120
BLUE_MAGIC_VALUE = 1.2


# ===================== FILTER FUNCTIONS =====================
def hue_shift_red(mat, h):
    U = math.cos(h * math.pi / 180)
    W = math.sin(h * math.pi / 180)

    r = (0.299 + 0.701 * U + 0.168 * W) * mat[..., 0]
    g = (0.587 - 0.587 * U + 0.330 * W) * mat[..., 1]
    b = (0.114 - 0.114 * U - 0.497 * W) * mat[..., 2]

    return np.dstack([r, g, b])


def normalizing_interval(array):
    high = 255
    low = 0
    max_dist = 0

    for i in range(1, len(array)):
        dist = array[i] - array[i - 1]
        if dist > max_dist:
            max_dist = dist
            high = array[i]
            low = array[i - 1]

    return (low, high)


def apply_filter(mat, filt):
    filtered_mat = np.zeros_like(mat, dtype=np.float32)
    filtered_mat[..., 0] = mat[..., 0] * filt[0] + mat[..., 1] * filt[1] + mat[..., 2] * filt[2] + filt[4] * 255
    filtered_mat[..., 1] = mat[..., 1] * filt[6] + filt[9] * 255
    filtered_mat[..., 2] = mat[..., 2] * filt[12] + filt[14] * 255
    return np.clip(filtered_mat, 0, 255).astype(np.uint8)


def get_filter_matrix(mat):
    mat = cv2.resize(mat, (256, 256))
    avg_mat = np.array(cv2.mean(mat)[:3], dtype=np.uint8)

    new_avg_r = avg_mat[0]
    hue_shift = 0

    while new_avg_r < MIN_AVG_RED:
        shifted = hue_shift_red(avg_mat, hue_shift)
        new_avg_r = np.sum(shifted)
        hue_shift += 1
        if hue_shift > MAX_HUE_SHIFT:
            break

    shifted_mat = hue_shift_red(mat, hue_shift)
    new_r_channel = np.sum(shifted_mat, axis=2)
    mat[..., 0] = np.clip(new_r_channel, 0, 255)

    hist_r = cv2.calcHist([mat], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([mat], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([mat], [2], None, [256], [0, 256])

    normalize_mat = np.zeros((256, 3))
    threshold = (mat.shape[0] * mat.shape[1]) / THRESHOLD_RATIO

    for x in range(256):
        if hist_r[x] < threshold:
            normalize_mat[x][0] = x
        if hist_g[x] < threshold:
            normalize_mat[x][1] = x
        if hist_b[x] < threshold:
            normalize_mat[x][2] = x

    normalize_mat[255] = [255, 255, 255]

    r_low, r_high = normalizing_interval(normalize_mat[..., 0])
    g_low, g_high = normalizing_interval(normalize_mat[..., 1])
    b_low, b_high = normalizing_interval(normalize_mat[..., 2])

    shifted = hue_shift_red(np.array([1, 1, 1]), hue_shift)
    sr, sg, sb = shifted[0][0]

    r_gain = 256 / (r_high - r_low)
    g_gain = 256 / (g_high - g_low)
    b_gain = 256 / (b_high - b_low)

    r_off = (-r_low / 256) * r_gain
    g_off = (-g_low / 256) * g_gain
    b_off = (-b_low / 256) * b_gain

    return np.array([
        sr * r_gain, sg * r_gain, sb * r_gain * BLUE_MAGIC_VALUE, 0, r_off,
        0, g_gain, 0, 0, g_off,
        0, 0, b_gain, 0, b_off,
        0, 0, 0, 1, 0,
    ])


def correct(mat):
    filt = get_filter_matrix(mat)
    corrected = apply_filter(mat, filt)
    return cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)


# ===================== MAIN =====================
def main():
    IMAGES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = [p for p in IMAGES_INPUT_DIR.iterdir() if p.suffix.lower() in valid_exts]

    print(f"{len(image_paths)} images found.")

    for img_path in image_paths:
        print(f"Processing: {img_path.name}")

        # Read image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print("  Failed to read image")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            filtered_bgr = correct(img_rgb)
        except Exception as e:
            print(f"  Filter error: {e}")
            continue

        # ---------- SAVE IMAGES ----------
        # original
        out_orig = IMAGES_OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_orig), img_bgr)

        # filtered
        filtered_name = img_path.stem + "_filtered" + img_path.suffix
        out_filt = IMAGES_OUTPUT_DIR / filtered_name
        cv2.imwrite(str(out_filt), filtered_bgr)

        # ---------- HANDLE LABELS ----------
        label_in = LABELS_INPUT_DIR / (img_path.stem + ".txt")

        if label_in.exists():
            # original label
            shutil.copy(label_in, LABELS_OUTPUT_DIR / label_in.name)

            # filtered label
            filtered_label = LABELS_OUTPUT_DIR / (img_path.stem + "_filtered.txt")
            shutil.copy(label_in, filtered_label)
        else:
            print("  Warning: label not found")

    print("Done.")


if __name__ == "__main__":
    main()