import cv2
import numpy as np
from pathlib import Path
from scripts.Blue_filter_V2 import *

# -------- CONFIG --------
IMAGE_DIR = Path(r"C:\Users\Xavier Lefebvre\Documents\dataset\Photo_Guillaueme\rgb-oakd")
LABEL_DIR = Path(r"C:\Users\Xavier Lefebvre\Documents\dataset\Photo_Guillaueme\label bbox")
OUTPUT_DIR = Path(r"C:\Users\Xavier Lefebvre\Documents\dataset\Photo_Guillaueme\Edge_detector")

IMAGE_EXTS = [".jpg", ".jpeg", ".png"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ajuste selon tes images
DARK_THRESHOLD = 170

# Kernel vertical pour poteaux
CLOSE_KERNEL_SIZE = (5, 15)
OPEN_KERNEL_SIZE = (3, 3)


def yolo_to_xyxy(label, img_w, img_h):
    class_id, xc, yc, w, h = label

    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    return int(class_id), x1, y1, x2, y2


def detect_dark_object_in_roi(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)

    # Seuil adaptatif : prend les pixels les plus foncés de la box
    adaptive_threshold = np.percentile(v, 35)

    threshold = min(DARK_THRESHOLD, adaptive_threshold)

    mask = cv2.inRange(v, 0, int(threshold))

    # Ferme les gaps sans trop favoriser vertical/horizontal
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    # Nettoyage léger
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filled = np.zeros_like(mask)

    if not contours:
        return filled

    h, w = mask.shape
    roi_area = h * w
    cx_roi = w / 2
    cy_roi = h / 2

    best_contour = None
    best_score = float("inf")

    for c in contours:
        area = cv2.contourArea(c)

        # Ignore bruit minuscule
        if area < 0.01 * roi_area:
            continue

        # Ignore un masque qui mange presque toute la box
        if area > 0.85 * roi_area:
            continue

        M = cv2.moments(c)

        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        dist_center = np.sqrt((cx - cx_roi) ** 2 + (cy - cy_roi) ** 2)

        # Favorise proche du centre + assez gros
        score = dist_center - 0.005 * area

        if score < best_score:
            best_score = score
            best_contour = c

    if best_contour is None:
        return filled

    cv2.drawContours(
        filled,
        [best_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    return filled

def detect_object_in_roi_V2(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Noir / foncé
    dark_threshold = min(DARK_THRESHOLD, np.percentile(v, 35))
    dark_mask = cv2.inRange(v, 0, int(dark_threshold))

    # Blanc / clair
    bright_threshold = max(200, np.percentile(v, 85))
    bright_mask = cv2.inRange(v, int(bright_threshold), 255)

    # Combine les deux
    mask = cv2.bitwise_or(dark_mask, bright_mask)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled = np.zeros_like(mask)

    if not contours:
        return filled

    biggest = max(contours, key=cv2.contourArea)

    cv2.drawContours(filled, [biggest], -1, 255, thickness=cv2.FILLED)

    return filled

def detect_object_in_roi(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # Valeur médiane du fond dans la box
    median_v = np.median(v)
    median_s = np.median(s)

    # Pixels très différents du fond
    diff_v = np.abs(v.astype(np.int16) - int(median_v)).astype(np.uint8)
    diff_s = np.abs(s.astype(np.int16) - int(median_s)).astype(np.uint8)

    mask = np.zeros_like(v)

    # Objet foncé OU objet clair OU saturation différente
    mask[
        (diff_v > 35) |
        (diff_s > 40)
    ] = 255

    # Morphologie
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 21))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filled = np.zeros_like(mask)

    if not contours:
        return filled

    biggest = max(contours, key=cv2.contourArea)

    cv2.drawContours(
        filled,
        [biggest],
        -1,
        255,
        thickness=cv2.FILLED
    )

    return filled

def process_image(image_path):
    img = cv2.imread(str(image_path))
    #img = blue_filter(img)

    # Rotation 180°
    #img = cv2.rotate(img, cv2.ROTATE_180)

    if img is None:
        print(f"Impossible de lire {image_path}")
        return

    img_h, img_w = img.shape[:2]

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        print(f"Aucun label pour {image_path.name}")
        return

    output_img = img.copy()

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        values = list(map(float, line.strip().split()))

        if len(values) != 5:
            continue

        class_id, x1, y1, x2, y2 = yolo_to_xyxy(values, img_w, img_h)

        roi = img[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        filled_mask = detect_dark_object_in_roi(roi)

        # Overlay blanc sur l'objet détecté
        output_img[y1:y2, x1:x2][filled_mask > 0] = [255, 255, 255]

        # Dessiner la box YOLO
        cv2.rectangle(
            output_img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output_img,
            f"id {class_id}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Rotation finale 180°
    output_img = cv2.rotate(output_img, cv2.ROTATE_180)

    save_path = OUTPUT_DIR / image_path.name
    cv2.imwrite(str(save_path), output_img)

    print(f"Sauvegardé: {save_path}")

def detect_object_in_roi_center_band(roi, band_width_ratio=0.35):
    h, w = roi.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    band_w = max(3, int(w * band_width_ratio))
    cx = w // 2

    x1 = max(0, cx - band_w // 2)
    x2 = min(w, cx + band_w // 2)

    mask[:, x1:x2] = 255

    return mask

def detect_object_in_roi_sobel(roi):

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Réduit le bruit
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Bords verticaux
    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1, 0,
        ksize=3
    )

    sobel_x = np.abs(sobel_x)

    # Normalisation 0-255
    sobel_x = cv2.normalize(
        sobel_x,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Seuil automatique
    thresh = np.percentile(sobel_x, 90)

    _, mask = cv2.threshold(
        sobel_x,
        thresh,
        255,
        cv2.THRESH_BINARY
    )

    # Relie les deux bords du poteau
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 25)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    # Nettoyage
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filled = np.zeros_like(mask)

    if not contours:
        return filled

    # Garde uniquement les contours verticaux
    valid_contours = []

    for c in contours:

        x, y, w, h = cv2.boundingRect(c)

        area = cv2.contourArea(c)

        if area < 20:
            continue

        ratio = h / max(w, 1)

        if ratio > 2:
            valid_contours.append(c)

    if not valid_contours:
        return filled

    biggest = max(
        valid_contours,
        key=cv2.contourArea
    )

    cv2.drawContours(
        filled,
        [biggest],
        -1,
        255,
        thickness=cv2.FILLED
    )

    return filled


def main():
    image_paths = []

    for ext in IMAGE_EXTS:
        image_paths.extend(IMAGE_DIR.glob(f"*{ext}"))

    for image_path in image_paths:
        process_image(image_path)


if __name__ == "__main__":
    main()