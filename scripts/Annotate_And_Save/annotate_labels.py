import cv2
import os
import glob
import yaml
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

# ---------------- CONFIG ----------------
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
IMAGE_DIR = Path(r"C:\Users\Xavier Lefebvre\Documents\dataset\rgb_oakd_14avril")
LABEL_DIR = Path(r"C:\Users\Xavier Lefebvre\Documents\dataset\labels_bbox_14avril")
DEPTH_DIR = REPO_ROOT / "datasets/Test_Piscine_a_annoter/Tests_march_18/depth"
IMAGE_EXT = ".jpg"
DATA_YAML = Path(r"C:\Users\Xavier Lefebvre\Documents\GitHub\NautilusVision\datasets\Test_Piscine_a_annoter\Tests_march_18\data_bbox.yaml")
START_INDEX = 0
ANNOTATION_MODE = "obb"  # "bbox" or "obb"

# ----------------------------------------

labels = []
click_points = []
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

def create_key_mapping_window(classes):
    key_mapping = {}

    root = tk.Tk()
    root.title("Association touches / classes")
    root.geometry("450x400")

    entries = {}

    tk.Label(
        root,
        text="Associer une touche à chaque classe",
        font=("Arial", 14, "bold")
    ).pack(pady=10)

    frame = tk.Frame(root)
    frame.pack(pady=10)

    for class_id, class_name in classes.items():
        row = tk.Frame(frame)
        row.pack(fill="x", pady=4)

        tk.Label(row, text=f"{class_id} - {class_name}", width=25, anchor="w").pack(side="left")

        entry = tk.Entry(row, width=5)
        entry.pack(side="left")
        entries[class_id] = entry

    def validate():
        used_keys = set()

        for class_id, entry in entries.items():
            key = entry.get().strip().lower()

            # Autoriser champ vide
            if key == "":
                continue

            #Toujours refuser multi-char
            if len(key) != 1:
                messagebox.showerror("Erreur", "Chaque touche doit être un seul caractère.")
                return

            #Bloquer touches interdites
            forbidden_keys = ['n', 'd','x', 'c', 'b', 'u', 's' ]
            if key in forbidden_keys:
                messagebox.showerror("Erreur", f"La touche '{key}' est interdite.")
                return

            #Empêcher doublons
            if key in used_keys:
                messagebox.showerror("Erreur", f"La touche '{key}' est utilisée plus d'une fois.")
                return

            used_keys.add(key)
            key_mapping[ord(key)] = class_id

        root.destroy()

    tk.Button(root, text="Valider", command=validate).pack(pady=15)

    root.mainloop()
    return key_mapping

KEY_TO_CLASS = create_key_mapping_window(CLASSES)

# ---------- LABEL IO ----------
def load_labels(label_path):
    if not os.path.exists(label_path):
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        open(label_path, "w").close()
        return []

    data = []
    with open(label_path, "r") as f:
        for line in f:
            p = line.strip().split()

            if ANNOTATION_MODE == "bbox" and len(p) == 5:
                data.append(list(map(float, p)))

            elif ANNOTATION_MODE == "obb" and len(p) == 9:
                data.append(list(map(float, p)))

    return data


def save_labels(label_path):
    with open(label_path, "w") as f:
        for l in labels:
            if ANNOTATION_MODE == "bbox":
                f.write(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")
            else:  # obb
                coords = " ".join(f"{v:.6f}" for v in l[1:])
                f.write(f"{int(l[0])} {coords}\n")
# ---------- DELETE ----------
def delete_nearest_label(x, y, max_dist_px=40):
    global labels

    best_i = None
    best_d = float("inf")

    for i, l in enumerate(labels):

        if ANNOTATION_MODE == "bbox":
            _, xc, yc, _, _ = l
            px = int(xc * img_w)
            py = int(yc * img_h)

        else:  # OBB
            pts = []
            for j in range(4):
                px = l[1 + 2*j] * img_w
                py = l[2 + 2*j] * img_h
                pts.append((px, py))

            # use center of polygon
            px = int(sum(p[0] for p in pts) / 4)
            py = int(sum(p[1] for p in pts) / 4)

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
    for l in labels:
        c = int(l[0])
        name = CLASSES.get(c, str(c))

        if ANNOTATION_MODE == "bbox":
            _, xc, yc, w, h = l

            x1 = int((xc - w / 2) * img_w - off_x)
            y1 = int((yc - h / 2) * img_h - off_y)
            x2 = int((xc + w / 2) * img_w - off_x)
            y2 = int((yc + h / 2) * img_h - off_y)

            x1 = int(x1 * zoom)
            y1 = int(y1 * zoom)
            x2 = int(x2 * zoom)
            y2 = int(y2 * zoom)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            text_pos = (x1, y1 - 5)

        else:  # OBB
            pts = []
            for i in range(4):
                px = l[1 + 2*i] * img_w
                py = l[2 + 2*i] * img_h

                px = int((px - off_x) * zoom)
                py = int((py - off_y) * zoom)

                pts.append((px, py))

            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 0, 255), 2)

            text_pos = (pts[0][0], pts[0][1] - 5)

        # ✅ Draw label text safely
        cv2.putText(
            img, name, text_pos,
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
    global click_points

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

        img_x = int(x / zoom + off_x)

        img_y = int(y / zoom + off_y)

        if ANNOTATION_MODE == "bbox":

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



        elif ANNOTATION_MODE == "obb":

            click_points.append((img_x, img_y))

            if len(click_points) == 3:

                import numpy as np

                p1 = np.array(click_points[0], dtype=float)

                p2 = np.array(click_points[1], dtype=float)

                p3 = np.array(click_points[2], dtype=float)

                # Direction vector (length of box)

                v = p2 - p1

                length = np.linalg.norm(v)

                if length == 0:
                    click_points = []

                    return

                v_unit = v / length

                # Perpendicular vector

                perp = np.array([-v_unit[1], v_unit[0]])

                # Width = projection of (p3 - p1) onto perpendicular

                width = np.dot(p3 - p1, perp)

                # Build rectangle corners

                p4 = p1 + perp * width

                p5 = p2 + perp * width

                pts = [p1, p2, p5, p4]  # ✅ CORRECT ORDER (no crossing)

                # Normalize

                norm_pts = []

                for px, py in pts:
                    norm_pts.append(px / img_w)

                    norm_pts.append(py / img_h)

                labels.append([current_class] + norm_pts)

                click_points = []

def delete_current_sample(img_path, label_path):
    try:
        # delete RGB image
        if os.path.exists(img_path):
            os.remove(img_path)

        # delete label
        if os.path.exists(label_path):
            os.remove(label_path)

        # delete depth map (same filename)
        depth_path = DEPTH_DIR / Path(img_path).name
        if os.path.exists(depth_path):
            os.remove(depth_path)

        print(f"Deleted: {img_path}")

    except Exception as e:
        print(f"Error deleting files: {e}")

# ---------- MAIN ----------
# ---------- MAIN ----------
image_files = sorted(IMAGE_DIR.glob(f"*{IMAGE_EXT}"))  # Path.glob returns Path objects

# ✅ FULLSCREEN WINDOW
cv2.namedWindow("Label Editor", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Label Editor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Label Editor", mouse_cb)

idx = START_INDEX if START_INDEX < len(image_files) else 0

while idx < len(image_files):
    img_path = image_files[idx]          # Path object
    base = img_path.stem                 # gets filename without extension
    label_path = LABEL_DIR / f"{base}.txt"

    img = cv2.imread(str(img_path))
    # cv2 needs string path
    if img is None:
        print(f"Warning: Could not read {img_path}")
        idx += 1
        continue

    img_h, img_w = img.shape[:2]
    zoom_cx, zoom_cy = img_w // 2, img_h // 2
    zoom = 1.0

    labels = load_labels(str(label_path))   # load_labels expects string path
    click_points = []

    while True:
        disp = get_zoom_view(img).copy()
        draw_labels(disp)
        draw_crosshair(disp, mouse_x, mouse_y)

        if drawing:
            xs = int((x_start - off_x) * zoom)
            ys = int((y_start - off_y) * zoom)
            cv2.rectangle(disp, (xs, ys), (mouse_x, mouse_y), (255, 0, 0), 1)

        if ANNOTATION_MODE == "obb" and len(click_points) > 0:
            for p in click_points:
                px = int((p[0] - off_x) * zoom)
                py = int((p[1] - off_y) * zoom)
                cv2.circle(disp, (px, py), 4, (255, 0, 0), -1)

        if ANNOTATION_MODE == "obb" and len(click_points) == 2:
            p1 = click_points[0]
            p2 = click_points[1]

            p1d = (int((p1[0] - off_x) * zoom), int((p1[1] - off_y) * zoom))
            p2d = (int((p2[0] - off_x) * zoom), int((p2[1] - off_y) * zoom))

            cv2.line(disp, p1d, p2d, (255, 0, 0), 2)

        info = f"{idx+1}/{len(image_files)} | Class {current_class}: {CLASSES[current_class]} | Zoom {zoom:.2f}x"
        cv2.putText(disp, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Label Editor", disp)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            cv2.destroyAllWindows()
            exit()
        elif key == ord('s'):
            save_labels(str(label_path))
        elif key == ord('n'):
            save_labels(str(label_path))
            idx += 1
            break
        elif key == ord("b"):
            save_labels(str(label_path))
            idx = max(0, idx - 1)
            break
        elif key == ord('c'):
            drawing = False
            click_points = []
        elif key == ord('d'):
            img_x = int(mouse_x / zoom + off_x)
            img_y = int(mouse_y / zoom + off_y)
            delete_nearest_label(img_x, img_y)
        elif key == ord('u') and labels:
            labels.pop()
        elif key == ord('x'):
            delete_current_sample(str(img_path), str(label_path))
            image_files = sorted(IMAGE_DIR.glob(f"*{IMAGE_EXT}"))
            break

        elif key in KEY_TO_CLASS:
            current_class = KEY_TO_CLASS[key]

cv2.destroyAllWindows()