import numpy as np
from pathlib import Path
import cv2
from scripts.Normalisation_depth import filter_depth


def detect_orange_boxes(img, min_area=900):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([0, 80, 80])
    upper_orange = np.array([25, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2

        boxes.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "cx": cx,
            "cy": cy,
            "area": area
        })

    return boxes, mask


def find_depth(image_path, kernel_size, half, use_filtered_depth=False):
    image_path = Path(image_path)
    depth_dir = Path(r"C:\Users\Xavier Lefebvre\Documents\dataset\depth")
    depth_path = depth_dir / image_path.name

    rgb = cv2.imread(str(image_path))
    if rgb is None:
        raise ValueError(f"Impossible de charger l'image RGB : {image_path}")

    h, w = rgb.shape[:2]

    # Détection orange
    boxes, mask = detect_orange_boxes(rgb)

    # Lecture de la depth
    if use_filtered_depth:
        depth = filter_depth(depth_path, kernel_size)
    else:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise ValueError(f"Impossible de charger l'image depth : {depth_path}")

    # Cas fréquent : (H, W, 1)
    depth = np.squeeze(depth)

    if len(depth.shape) != 2:
        raise ValueError(f"Depth non 2D après squeeze : {depth.shape}")

    results = []
    output = rgb.copy()

    for i, box in enumerate(boxes):
        x = box["x"]
        y = box["y"]
        bw = box["w"]
        bh = box["h"]
        x_center = box["cx"]
        y_center = box["cy"]

        # Patch autour du centre pour estimer la profondeur
        x1 = max(0, x_center - half)
        x2 = min(w, x_center + half)
        y1 = max(0, y_center - half)
        y2 = min(h, y_center + half)

        patch = depth[y1:y2, x1:x2]

        if patch.size == 0:
            print(f"Box {i}: patch vide")
            continue

        # Ignore les zéros si depth invalide
        valid_patch = patch[patch > 0]
        if valid_patch.size == 0:
            print(f"Box {i}: aucune profondeur valide")
            continue

        depth_mean = float(np.median(valid_patch))

        angle = find_angle(
            x_center=x_center,
            depth_mean=depth_mean,
        )

        results.append({
            "box_id": i,
            "x": x,
            "y": y,
            "w": bw,
            "h": bh,
            "cx": x_center,
            "cy": y_center,
            "depth": depth_mean,
            "angle_deg": angle
        })

        # Dessin
        cv2.rectangle(output, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.circle(output, (x_center, y_center), 3, (0, 0, 255), -1)
        cv2.putText(
            output,
            f"z={depth_mean:.1f}  yaw={angle:.1f} deg",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        print(
            f"Box {i} | centre=({x_center},{y_center}) | "
            f"profondeur={depth_mean:.2f} | angle={angle:.2f} deg"
        )

    return results, output, mask


def find_angle(x_center, depth_mean):
    # Coordonnée horizontale dans le repère caméra
    fx = 728
    cx = 640

    X = (x_center - cx) * depth_mean / fx

    # Yaw pour aligner l'objet avec le centre
    yaw = np.arctan2(X, depth_mean)
    yaw_deg = np.degrees(yaw)

    return yaw_deg

if __name__ == "__main__":
    RGB_DIR = Path(r"C:\Users\Xavier Lefebvre\Documents\dataset\rgb")
    OUTPUT_DIR = RGB_DIR.parent / "rgb_processed_2"
    OUTPUT_DIR.mkdir(exist_ok=True)

    valid_exts = [".png", ".jpg", ".jpeg"]

    for image_path in RGB_DIR.iterdir():

        if image_path.suffix.lower() not in valid_exts:
            continue

        print(f"\nProcessing: {image_path.name}")

        try:
            results, output, mask = find_depth(
                image_path=image_path,
                kernel_size=5,
                half=5,
                use_filtered_depth=False,
            )

            # Sauvegarde image résultat
            out_path = OUTPUT_DIR / image_path.name
            cv2.imwrite(str(out_path), output)

        except Exception as e:
            print(f"Erreur avec {image_path.name}: {e}")

    print("\nDone ✅")