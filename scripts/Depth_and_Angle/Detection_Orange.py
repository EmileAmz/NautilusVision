import numpy as np
import cv2

from scripts.Depth_and_Angle.Depth_and_angle import find_depth


def detect_orange_boxes(img, depth_frame, min_area=900):
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

        depth = find_depth(
            depth_frame=depth_frame,
            half=5,
            cy_boundingbox=cy,
            cx_boundingbox=cx,
            kernel_size=5
        )

        box = {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "cx": cx,
            "cy": cy,
            "area": area,
            "depth": depth
        }

        boxes.append(box)

    return boxes, mask


def draw_orange_boxes(frame, boxes):
    output = frame.copy()

    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        cx, cy = box["cx"], box["cy"]

        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(output, (cx, cy), 3, (0, 0, 255), -1)

        depth = box["depth"]
        label = f"{depth}mm"

        cv2.putText(
            output,
            label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    return output