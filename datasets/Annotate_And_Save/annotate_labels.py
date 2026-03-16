import cv2
import os
import glob
import yaml

# ---------------- CONFIG ----------------
IMAGE_DIR = "Data/train/images"
LABEL_DIR = "Data/train/labels"
IMAGE_EXT = ".jpg"
DATA_YAML = "Data/data.yaml"
# ----------------------------------------

labels = []
img_w = img_h = 1
current_class = 0

drawing = False
x_start = y_start = 0
mouse_x = mouse_y = 0

# ---------- ZOOM ----------
zoom = 1.0
ZOOM_MIN = 1.0
ZOOM_MAX = 8.0
zoom_cx = 0
zoom_cy = 0
off_x = off_y = 0

# ---------- LOAD CLASSES ----------
def load_classes_from_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data["names"]
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


CLASSES = load_classes_from_yaml(DATA_YAML)

# ---------- LABEL IO ----------
def load_labels(label_path):
    data = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) == 5:
                    data.append(list(map(float, p)))
    return data


def save_labels(label_path):
    with open(label_path, "w") as f:
        for l in labels:
            f.write(
                f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n"
            )

# ---------- DELETE ----------
def delete_nearest_label(x, y, max_dist_px=40):
    global labels

    best_i = None
    best_d = float("inf")

    for i, (_, xc, yc, _, _) in enumerate(labels):
        px = int(xc * img_w)
        py = int(yc * img_h)
        d = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
        if d < best_d and d < max_dist_px:
            best_d = d
            best_i = i

    if best_i is not None:
        del labels[best_i]

# ---------- DRAW ----------
def draw_crosshair(img, x, y):
    cv2.line(img, (x, 0), (x, img.shape[0]), (0, 255, 0), 1)
    cv2.line(img, (0, y), (img.shape[1], y), (0, 255, 0), 1)


def draw_labels(img):
    for c, xc, yc, w, h in labels:
        x1 = int((xc - w / 2) * img_w - off_x)
        y1 = int((yc - h / 2) * img_h - off_y)
        x2 = int((xc + w / 2) * img_w - off_x)
        y2 = int((yc + h / 2) * img_h - off_y)

        x1 = int(x1 * zoom)
        y1 = int(y1 * zoom)
        x2 = int(x2 * zoom)
        y2 = int(y2 * zoom)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        name = CLASSES.get(int(c), str(c))
        cv2.putText(
            img, name, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )

# ---------- ZOOM VIEW ----------
def get_zoom_view(img):
    global off_x, off_y

    if zoom == 1.0:
        off_x = off_y = 0
        return img

    h, w = img.shape[:2]
    vw = int(w / zoom)
    vh = int(h / zoom)

    x1 = max(0, min(zoom_cx - vw // 2, w - vw))
    y1 = max(0, min(zoom_cy - vh // 2, h - vh))

    off_x, off_y = x1, y1

    crop = img[y1:y1 + vh, x1:x1 + vw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

# ---------- MOUSE ----------
def mouse_cb(event, x, y, flags, param):
    global drawing, x_start, y_start
    global mouse_x, mouse_y
    global zoom, zoom_cx, zoom_cy

    mouse_x, mouse_y = x, y

    img_x = int(x / zoom + off_x)
    img_y = int(y / zoom + off_y)

    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            zoom = min(ZOOM_MAX, zoom * 1.25)
        else:
            zoom = max(ZOOM_MIN, zoom / 1.25)

        zoom_cx = img_x
        zoom_cy = img_y

    elif event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            x_start, y_start = img_x, img_y
            drawing = True
        else:
            x_end, y_end = img_x, img_y
            drawing = False

            x1, x2 = sorted([x_start, x_end])
            y1, y2 = sorted([y_start, y_end])

            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            xc = ((x1 + x2) / 2) / img_w
            yc = ((y1 + y2) / 2) / img_h

            if w > 0 and h > 0:
                labels.append([current_class, xc, yc, w, h])

# ---------- MAIN ----------
image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*" + IMAGE_EXT)))

# ✅ FULLSCREEN WINDOW
cv2.namedWindow("Label Editor", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Label Editor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.setMouseCallback("Label Editor", mouse_cb)

idx = 0
while idx < len(image_files):
    img_path = image_files[idx]
    base = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(LABEL_DIR, base + ".txt")

    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    zoom_cx, zoom_cy = img_w // 2, img_h // 2
    zoom = 1.0

    labels = load_labels(label_path)

    while True:
        disp = get_zoom_view(img).copy()
        draw_labels(disp)
        draw_crosshair(disp, mouse_x, mouse_y)

        if drawing:
            xs = int((x_start - off_x) * zoom)
            ys = int((y_start - off_y) * zoom)
            cv2.rectangle(disp, (xs, ys), (mouse_x, mouse_y), (255, 0, 0), 1)

        info = f"{idx+1}/{len(image_files)} | Class {current_class}: {CLASSES[current_class]} | Zoom {zoom:.2f}x"
        cv2.putText(disp, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Label Editor", disp)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            cv2.destroyAllWindows()
            exit()
        elif key == ord('s'):
            save_labels(label_path)
        elif key == ord('n'):
            save_labels(label_path)
            idx += 1
            break
        elif key == ord('c'):
            drawing = False
        elif key == ord('d'):
            img_x = int(mouse_x / zoom + off_x)
            img_y = int(mouse_y / zoom + off_y)
            delete_nearest_label(img_x, img_y)
        elif key == ord('u') and labels:
            labels.pop()
        elif ord('0') <= key <= ord('9'):
            cid = key - ord('0')
            if cid in CLASSES:
                current_class = cid

cv2.destroyAllWindows()