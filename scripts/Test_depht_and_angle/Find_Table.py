import cv2
import numpy as np
from pathlib import Path
from Test_depth_and_angle import find_angle, find_depth


def load_yolo_boxes(txt_path, img_width, img_height):
    boxes = []

    txt_path = Path(txt_path)

    # Dossier où se trouvent les images depth
    depth_dir = Path(r"C:\Users\Xavier Lefebvre\Documents\dataset\depth")

    # Prendre le nom du fichier et changer l'extension
    depth_name = txt_path.stem + ".png"
    depth_path = depth_dir / depth_name

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        INPUT_SIZE = 416  # ou 224, 416, etc selon ton modèle

        cx = int(x_center * INPUT_SIZE)
        cy = int(y_center * INPUT_SIZE)
        w = int(width * INPUT_SIZE)
        h = int(height * INPUT_SIZE)

        # Puis tu rescales vers l'image réelle
        scale_x = img_width / INPUT_SIZE
        scale_y = img_height / INPUT_SIZE

        # Conversion YOLO -> pixels
        cx = int(cx * scale_x) +20
        cy = int(cy * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)

        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        box = {
            "class_id": class_id,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }

        depth = find_depth(
            depth_path=depth_path,
            kernel_size=5,
            half=25,
            box=box,
            image_width=img_width,
            image_height=img_height,
            use_filtered_depth=True
        )
        angle = find_angle(box, depth)

        box["depth"] = depth
        box["angle"] = angle

        boxes.append(box)

    return boxes


def draw_boxes_and_centers(img, boxes, angle_gate):
    output = img.copy()

    for box in boxes:
        x1, y1 = box["x1"], box["y1"]
        x2, y2 = box["x2"], box["y2"]
        cx, cy = box["cx"], box["cy"]
        class_id = box["class_id"]
        depth = box.get("depth")
        angle = box.get("angle")

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)

        label1 = f"id={class_id} center=({cx},{cy})"
        cv2.putText(
            output,
            label1,
            (x1, max(20, y1 - 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        if depth is not None and angle is not None:
            label2 = f"depth={depth:.1f} angle={angle:.1f}"
        else:
            label2 = "depth/angle invalide"

        cv2.putText(
            output,
            label2,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Affichage angle global de la gate
    cv2.putText(
        output,
        f"angle_gate={angle_gate:.2f} deg" if angle_gate is not None else "angle_gate invalide",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    return output


def find_angle_plane(boxes):
    profondeurs = []
    angles = []
    C = 1442
    phi = 0

    for box in boxes:
        profondeurs.append(box["depth"])
        angles.append(box["angle"])

    if angles[0] <= 0 and angles[1] <= 0:

        if np.abs(angles[0]) > np.abs(angles[1]):
            teta_g = np.abs(angles[0])
            A = profondeurs[0]

            alpha = np.abs(angles[1])
            B = profondeurs[1]

        else:
            teta_g = np.abs(angles[1])
            A = profondeurs[1]

            alpha = np.abs(angles[0])
            B = profondeurs[0]

        teta_c = teta_g - alpha
        phi = 90 - alpha
        teta_c_rad = np.radians(teta_c)
        teta_a_rad = np.arcsin((np.sin(teta_c_rad) * A) / C)
        teta_a = np.degrees(teta_a_rad)
        result = 180 - teta_a - phi

    elif angles[0] > 0 and angles[1] > 0:
        if np.abs(angles[0]) > np.abs(angles[1]):
            teta_g = np.abs(angles[1])
            A = profondeurs[1]

            alpha = np.abs(angles[0])
            B = profondeurs[0]

        else:
            teta_g = np.abs(angles[0])
            A = profondeurs[0]

            alpha = np.abs(angles[1])
            B = profondeurs[1]

        teta_c = teta_g - alpha
        phi = 90 - alpha
        teta_c_rad = np.radians(teta_c)
        teta_a_rad = np.arcsin((np.sin(teta_c_rad) * A) / C)
        teta_a = np.degrees(teta_a_rad)
        result = 180 - teta_a - phi


    else:
        if np.abs(angles[0]) > np.abs(angles[1]):
            teta_g = np.abs(angles[0])
            A = profondeurs[0]

            alpha = np.abs(angles[1])
            B = profondeurs[1]

        else:
            teta_g = np.abs(angles[1])
            A = profondeurs[1]

            alpha = np.abs(angles[0])
            B = profondeurs[0]

        teta_c = teta_g + alpha
        phi = alpha
        teta_c_rad = np.radians(teta_c)
        teta_a_rad = np.arcsin((np.sin(teta_c_rad) * A) / C)
        teta_a = np.degrees(teta_a_rad)
        result = 90 - teta_a - phi

    return result


if __name__ == "__main__":
    img_path = Path(r"C:\Users\Xavier Lefebvre\Documents\dataset\Gate_table\1775152864.904.png")
    txt_path = Path(r"C:\Users\Xavier Lefebvre\Documents\dataset\labels_bbox\1775152864.904.txt")

    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Impossible de charger l'image: {img_path}")

    img_height, img_width = img.shape[:2]

    boxes = load_yolo_boxes(txt_path, img_width, img_height)
    angle_gate = find_angle_plane(boxes)

    for i, box in enumerate(boxes):
        print(
            f"Box {i}: class={box['class_id']}, "
            f"center=({box['cx']}, {box['cy']}), "
            f"size=({box['w']}, {box['h']}), "
            f"depth={box['depth']}, "
            f"angle={box['angle']}"
        )

    result = draw_boxes_and_centers(img, boxes, angle_gate)

    cv2.imshow("Boxes and centers", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()