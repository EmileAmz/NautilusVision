import numpy as np
import cv2
from pathlib import Path
from Test_depth_and_angle import find_angle, find_depth


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


if __name__ == "__main__":
    RGB_DIR = Path(r"C:\Users\Xavier Lefebvre\Documents\dataset\rgb")
    DEPTH_DIR = Path(r"C:\Users\Xavier Lefebvre\Documents\dataset\depth")
    OUTPUT_DIR = RGB_DIR.parent / "rgb_processed"
    OUTPUT_DIR.mkdir(exist_ok=True)

    valid_exts = [".png", ".jpg", ".jpeg"]
    results = []

    for image_path in RGB_DIR.iterdir():
        if image_path.suffix.lower() not in valid_exts:
            continue

        print(f"\nProcessing: {image_path.name}")

        rgb = cv2.imread(str(image_path))
        if rgb is None:
            print(f"Impossible de charger l'image RGB : {image_path}")
            continue

        h, w = rgb.shape[:2]

        # Construire le chemin depth correspondant
        depth_path = DEPTH_DIR / (image_path.stem + ".png")

        try:
            boxes, mask = detect_orange_boxes(rgb)

            output = rgb.copy()

            for i, box in enumerate(boxes):
                depth = find_depth(
                    depth_path=depth_path,
                    kernel_size=5,
                    half=5,
                    box=box,
                    image_width=w,
                    image_height=h,
                    use_filtered_depth=False
                )

                angle = find_angle(box, depth)

                result = {
                    "box_id": i,
                    "x": box["x"],
                    "y": box["y"],
                    "w": box["w"],
                    "h": box["h"],
                    "cx": box["cx"],
                    "cy": box["cy"],
                    "depth": depth,
                    "angle_deg": angle
                }
                results.append(result)

                # Affichage seulement si depth valide
                label = f"z={depth:.1f} yaw={angle:.1f} deg" if depth is not None and angle is not None else "depth invalide"

                cv2.rectangle(
                    output,
                    (box["x"], box["y"]),
                    (box["x"] + box["w"], box["y"] + box["h"]),
                    (0, 255, 0),
                    2
                )
                cv2.circle(output, (box["cx"], box["cy"]), 3, (0, 0, 255), -1)
                cv2.putText(
                    output,
                    label,
                    (box["x"], max(20, box["y"] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                print(
                    f"Box {i} | centre=({box['cx']},{box['cy']}) | "
                    f"profondeur={depth} | angle={angle}"
                )

            out_path = OUTPUT_DIR / image_path.name
            cv2.imwrite(str(out_path), output)

        except Exception as e:
            print(f"Erreur avec {image_path.name}: {e}")

    print("\nDone ✅")