import cv2
import numpy as np


def detect_orange_boxes(img, depth_frame, min_area=900):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([0, 80, 80])
    upper_orange = np.array([20, 255, 255])

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
        depth = find_depth(depth_frame, 5, cy, cx)
        angle = find_angle(cx, depth)


        boxes.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "cx": cx,
            "cy": cy,
            "area": area,
            "angle": angle,
            "depth": depth
        })

    return boxes, mask


def draw_orange_boxes(img, boxes):
    output = img.copy()

    for box in boxes:
        x = box["x"]
        y = box["y"]
        w = box["w"]
        h = box["h"]
        cx = box["cx"]
        cy = box["cy"]
        area = int(box["area"])
        profondeur = box["depth"]
        angle = box["angle"]

        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(output, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(
            output,
            f"Profondeur = {profondeur}"
            f"Angle = {angle}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return output

def find_depth(depth_frame, half, cy, cx):
    frame_h, frame_w = depth_frame.shape

    y1 = max(0, cy - half)
    y2 = min(frame_h, cy + half + 1)
    x1 = max(0, cx - half)
    x2 = min(frame_w, cx + half + 1)

    center_patch = depth_frame[y1:y2, x1:x2]

    valid_center = center_patch[(center_patch > 200) & (center_patch < 5000)]

    if valid_center.size > 0:
        center_depth = float(np.median(valid_center))
    else:
        print("Center depth: invalid")
        center_depth = None

    return center_depth


def find_angle(x_center, depth_mean):
    # Coordonnée horizontale dans le repère caméra
    fx = 728
    cx = 640

    if depth_mean is None:
        return None

    X = (x_center - cx) * depth_mean / fx

    # Yaw pour aligner l'objet avec le centre
    yaw = np.arctan2(X, depth_mean)
    yaw_deg = np.degrees(yaw)

    return yaw_deg