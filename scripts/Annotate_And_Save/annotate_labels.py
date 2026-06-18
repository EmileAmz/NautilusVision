import cv2
import os
import yaml
import json
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# =====================================================
# PATHS
# =====================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
YAML_DIR = SCRIPT_DIR / "YAML"
SETTINGS_FILE = YAML_DIR / "last_selection.txt"

# =====================================================
# GLOBALS
# =====================================================
labels = []
click_points = []

img_w = img_h = 1
current_class = 0

drawing = False
x_start = y_start = 0
mouse_x = mouse_y = 0

zoom = 1.0
ZOOM_MIN = 1.0
ZOOM_MAX = 8.0
zoom_cx = 0
zoom_cy = 0
off_x = off_y = 0

IMAGE_DIR = None
LABEL_DIR = None
DEPTH_DIR = None
DATA_YAML = None
ANNOTATION_MODE = "bbox"
START_INDEX = 0

CLASSES = {}
KEY_TO_CLASS = {}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# =====================================================
# YAML
# =====================================================
def load_classes_from_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data["names"]

    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}

    return {int(k): v for k, v in names.items()}



# =====================================================
# SETTINGS
# =====================================================
def load_last_settings():
    if not SETTINGS_FILE.exists():
        return {}

    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_last_settings(settings):
    YAML_DIR.mkdir(parents=True, exist_ok=True)

    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4)

# =====================================================
# STARTUP GUI
# =====================================================
def startup_window():
    global IMAGE_DIR, LABEL_DIR, DEPTH_DIR
    global DATA_YAML, ANNOTATION_MODE, START_INDEX, CLASSES, KEY_TO_CLASS

    config = {
        "dataset_dir": None,
        "yaml_path": None,
        "mode": "bbox",
        "start_index": 0,
        "key_mapping": {}
    }

    last_settings = load_last_settings()

    root = tk.Tk()
    root.title("Configuration Label Editor")

    window_w = 700
    window_h = 760
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    x = (screen_w - window_w) // 2
    y = (screen_h - window_h) // 2
    root.geometry(f"{window_w}x{window_h}+{x}+{y}")
    root.minsize(window_w, window_h)

    dataset_var = tk.StringVar(value=last_settings.get("dataset_dir", ""))
    yaml_var = tk.StringVar()
    mode_var = tk.StringVar(value=last_settings.get("mode", "bbox"))
    start_index_var = tk.StringVar(value=str(last_settings.get("start_index", 0)))

    entries = {}

    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)

    left_frame = tk.Frame(main_frame)
    left_frame.pack(side="left", fill="both", expand=True)

    right_frame = tk.Frame(main_frame)
    right_frame.pack(side="right", fill="y", padx=(20, 0))

    class_frame = tk.Frame(right_frame)

    YAML_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- DATASET ----------------
    def choose_dataset():
        folder = filedialog.askdirectory(title="Choisir le dossier du dataset")
        if folder:
            dataset_var.set(folder)

    tk.Label(left_frame, text="1. Dataset", font=("Arial", 13, "bold")).pack(pady=(15, 5))

    dataset_row = tk.Frame(left_frame)
    dataset_row.pack(fill="x", padx=20)

    tk.Entry(dataset_row, textvariable=dataset_var).pack(side="left", fill="x", expand=True)
    tk.Button(dataset_row, text="Choisir", command=choose_dataset).pack(side="left", padx=5)

    # ---------------- YAML ----------------
    tk.Label(left_frame, text="2. YAML", font=("Arial", 13, "bold")).pack(pady=(20, 5))

    yaml_files = sorted(list(YAML_DIR.glob("*.yaml")) + list(YAML_DIR.glob("*.yml")))
    yaml_names = [p.name for p in yaml_files]
    yaml_map = {p.name: p for p in yaml_files}

    last_yaml_name = last_settings.get("yaml_name", "")

    if last_yaml_name in yaml_names:
        yaml_var.set(last_yaml_name)
    elif yaml_names:
        yaml_var.set(yaml_names[0])

    yaml_combo = ttk.Combobox(
        root,
        textvariable=yaml_var,
        values=yaml_names,
        state="readonly",
        width=50
    )
    yaml_combo.pack(in_=left_frame, padx=20)

    yaml_combo.bind(
        "<<ComboboxSelected>>",
        lambda event: refresh_classes()
    )

    # ---------------- MODE ----------------
    tk.Label(left_frame, text="3. Mode d'annotation", font=("Arial", 13, "bold")).pack(pady=(20, 5))

    mode_row = tk.Frame(left_frame)
    mode_row.pack()

    tk.Radiobutton(mode_row, text="BBox", variable=mode_var, value="bbox").pack(side="left", padx=10)
    tk.Radiobutton(mode_row, text="OBB", variable=mode_var, value="obb").pack(side="left", padx=10)

    # ---------------- START INDEX ----------------
    tk.Label(left_frame, text="4. Start index", font=("Arial", 13, "bold")).pack(pady=(20, 5))

    tk.Entry(left_frame, textvariable=start_index_var, width=10).pack()

    # ---------------- KEY MAPPING ----------------
    tk.Label(right_frame, text="Assignation des touches", font=("Arial", 13, "bold")).pack(pady=(20, 5))

    class_frame.pack(fill='y', pady=5)

    def refresh_classes():
        nonlocal entries

        for widget in class_frame.winfo_children():
            widget.destroy()

        entries = {}

        yaml_path = yaml_map.get(yaml_var.get())

        if yaml_path is None:
            messagebox.showerror("Erreur", "Aucun fichier YAML trouvé dans le dossier YAML.")
            return

        try:
            classes = load_classes_from_yaml(yaml_path)
        except Exception as e:
            messagebox.showerror("Erreur YAML", str(e))
            return

        for class_id, class_name in classes.items():
            row = tk.Frame(class_frame)
            row.pack(fill="x", pady=3)

            tk.Label(row, text=f"{class_id} - {class_name}", width=35, anchor="w").pack(side="left")

            entry = tk.Entry(row, width=5)
            entry.pack(side="left")

            saved_keys = last_settings.get("keys", {})
            saved_key = saved_keys.get(str(class_id), "")
            if saved_key:
                entry.insert(0, saved_key)

            entries[class_id] = entry

    # ---------------- VALIDATION ----------------
    def validate():
        dataset_path = Path(dataset_var.get())
        yaml_path = yaml_map.get(yaml_var.get())

        if not dataset_path.exists():
            messagebox.showerror("Erreur", "Le dossier dataset n'existe pas.")
            return

        if yaml_path is None or not yaml_path.exists():
            messagebox.showerror("Erreur", "Le fichier YAML n'existe pas.")
            return

        try:
            start_index = int(start_index_var.get())
            if start_index < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Erreur", "Start index doit être un entier positif.")
            return

        if (dataset_path / "images").exists():
            image_dir = dataset_path / "images"
        else:
            image_dir = dataset_path

        label_dir = dataset_path / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)

        depth_dir = dataset_path / "depth"

        try:
            classes = load_classes_from_yaml(yaml_path)
        except Exception as e:
            messagebox.showerror("Erreur YAML", str(e))
            return

        forbidden_keys = ["n", "b", "d", "c", "u", "s", "q"]
        used_keys = set()
        key_mapping = {}
        key_mapping_save = {}

        for class_id, entry in entries.items():
            key = entry.get().strip().lower()

            if key == "":
                continue

            if len(key) != 1:
                messagebox.showerror("Erreur", "Chaque touche doit être un seul caractère.")
                return

            if key in forbidden_keys:
                messagebox.showerror("Erreur", f"La touche '{key}' est réservée.")
                return

            if key in used_keys:
                messagebox.showerror("Erreur", f"La touche '{key}' est utilisée plus d'une fois.")
                return

            used_keys.add(key)
            key_mapping[ord(key)] = class_id
            key_mapping_save[str(class_id)] = key

        if not key_mapping:
            messagebox.showerror("Erreur", "Tu dois assigner au moins une touche.")
            return

        config["dataset_dir"] = dataset_path
        config["image_dir"] = image_dir
        config["label_dir"] = label_dir
        config["depth_dir"] = depth_dir
        config["yaml_path"] = yaml_path
        config["mode"] = mode_var.get()
        config["start_index"] = start_index
        config["classes"] = classes
        config["key_mapping"] = key_mapping

        save_last_settings({
            "dataset_dir": str(dataset_path),
            "yaml_name": yaml_path.name,
            "mode": mode_var.get(),
            "start_index": start_index,
            "keys": key_mapping_save
        })

        root.destroy()

    tk.Button(left_frame, text="Démarrer l'annotation", command=validate, height=2).pack(pady=25)

    refresh_classes()
    root.mainloop()

    if config["dataset_dir"] is None:
        exit()

    IMAGE_DIR = config["image_dir"]
    LABEL_DIR = config["label_dir"]
    DEPTH_DIR = config["depth_dir"]
    DATA_YAML = config["yaml_path"]
    ANNOTATION_MODE = config["mode"]
    START_INDEX = config["start_index"]
    CLASSES = config["classes"]
    KEY_TO_CLASS = config["key_mapping"]


# =====================================================
# LABEL IO
# =====================================================
def load_labels(label_path):
    if not label_path.exists():
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.touch()
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
            else:
                coords = " ".join(f"{v:.6f}" for v in l[1:])
                f.write(f"{int(l[0])} {coords}\n")


# =====================================================
# DELETE
# =====================================================
def delete_nearest_label(x, y, max_dist_px=40):
    global labels

    best_i = None
    best_d = float("inf")

    for i, l in enumerate(labels):
        if ANNOTATION_MODE == "bbox":
            _, xc, yc, _, _ = l
            px = int(xc * img_w)
            py = int(yc * img_h)

        else: #OBB
            pts = []
            for j in range(4):
                px = l[1 + 2 * j] * img_w
                py = l[2 + 2 * j] * img_h
                pts.append((px, py))

            px = int(sum(p[0] for p in pts) / 4)
            py = int(sum(p[1] for p in pts) / 4)

        d = ((px - x) ** 2 + (py - y) ** 2) ** 0.5

        if d < best_d and d < max_dist_px:
            best_d = d
            best_i = i

    if best_i is not None:
        del labels[best_i]


# =====================================================
# DRAW
# =====================================================
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

        else:
            pts = []

            for i in range(4):
                px = l[1 + 2 * i] * img_w
                py = l[2 + 2 * i] * img_h

                px = int((px - off_x) * zoom)
                py = int((py - off_y) * zoom)

                pts.append((px, py))

            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 0, 255), 2)
            text_pos = (pts[0][0], pts[0][1] - 5)

        cv2.putText(
            img,
            name,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )


def draw_help(img, idx, total, filename):
    key_text = []

    for key_code, class_id in KEY_TO_CLASS.items():
        key_text.append(f"{chr(key_code)}:{CLASSES[class_id]}")

    text_1 = f"{idx + 1}/{total} | Classe {current_class}: {CLASSES[current_class]} | Zoom {zoom:.2f}x"
    text_0 = filename
    #text_2 = "n:next | b:back | s:save | d:delete | u:undo | c:cancel | x:delete image | ESC:quit"
    #text_3 = "Classes: " + " | ".join(key_text)

    cv2.putText(
        img,
        text_0,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        img,
        text_1,
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    #cv2.putText(img, text_2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    #cv2.putText(img, text_3, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


# =====================================================
# ZOOM
# =====================================================
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


# =====================================================
# MOUSE
# =====================================================
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
                p1 = np.array(click_points[0], dtype=float)
                p2 = np.array(click_points[1], dtype=float)
                p3 = np.array(click_points[2], dtype=float)

                v = p2 - p1
                length = np.linalg.norm(v)

                if length == 0:
                    click_points = []
                    return

                v_unit = v / length
                perp = np.array([-v_unit[1], v_unit[0]])
                width = np.dot(p3 - p1, perp)

                p4 = p1 + perp * width
                p5 = p2 + perp * width

                pts = [p1, p2, p5, p4]

                norm_pts = []

                for px, py in pts:
                    norm_pts.append(px / img_w)
                    norm_pts.append(py / img_h)

                labels.append([current_class] + norm_pts)
                click_points = []


# =====================================================
# IMAGE FILES
# =====================================================
def get_image_files():
    files = []

    for p in IMAGE_DIR.iterdir():
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)

    return sorted(files)


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    startup_window()

    image_files = get_image_files()

    if len(image_files) == 0:
        print(f"Aucune image trouvée dans: {IMAGE_DIR}")
        exit()

    cv2.namedWindow("Label Editor", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Label Editor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("Label Editor", mouse_cb)

    idx = START_INDEX if START_INDEX < len(image_files) else 0

    while idx < len(image_files):
        img_path = image_files[idx]
        label_path = LABEL_DIR / f"{img_path.stem}.txt"

        img = cv2.imread(str(img_path))

        if img is None:
            print(f"Warning: Could not read {img_path}")
            idx += 1
            continue

        img_h, img_w = img.shape[:2]
        zoom_cx, zoom_cy = img_w // 2, img_h // 2
        zoom = 1.0

        labels = load_labels(label_path)
        click_points = []
        drawing = False

        while True:
            disp = get_zoom_view(img).copy()

            draw_labels(disp)
            draw_crosshair(disp, mouse_x, mouse_y)
            draw_help(disp, idx, len(image_files), img_path.name)

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

            cv2.imshow("Label Editor", disp)
            key = cv2.waitKey(20) & 0xFF

            if key == 27:
                cv2.destroyAllWindows()
                exit()

            elif key == ord("s"):
                save_labels(label_path)
                print(f"Saved: {label_path}")

            elif key == ord("n"):
                save_labels(label_path)
                idx += 1
                break

            elif key == ord("b"):
                save_labels(label_path)
                idx = max(0, idx - 1)
                break

            elif key == ord("c"):
                drawing = False
                click_points = []

            elif key == ord("d"):
                img_x = int(mouse_x / zoom + off_x)
                img_y = int(mouse_y / zoom + off_y)
                delete_nearest_label(img_x, img_y)

            elif key == ord("u") and labels:
                labels.pop()
            

            elif key in KEY_TO_CLASS:
                current_class = KEY_TO_CLASS[key]
                print(f"Selected class: {current_class} - {CLASSES[current_class]}")

    cv2.destroyAllWindows()