#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mini ROS2 bag viewer WITHOUT ROS2 environment.

Works on Windows/Linux with:
    pip install matplotlib zstandard pillow

Then:
    python mini_rosbag_viewer_select_paths.py

You can select separately:
- the rosbag folder/file
- the annotated image folder

Supported message decoding:
- std_msgs/msg/Int32
- std_msgs/msg/Float32
- std_msgs/msg/Float64
- std_msgs/msg/String
- std_msgs/msg/Int32MultiArray
- std_msgs/msg/Float32MultiArray
- std_msgs/msg/Float64MultiArray
- geometry_msgs/msg/Vector3
- geometry_msgs/msg/Twist
- geometry_msgs/msg/TwistStamped

For unsupported/custom messages, it shows raw hex preview.

Important:
If your bag was recorded with:
    --compression-mode file --compression-format zstd
the .db3 may be compressed as .db3.zstd. This script auto-decompresses it to a temp .db3 copy.
"""

import os
import re
import sys
import math
import sqlite3
import struct
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import tkinter as tk
from tkinter import filedialog, ttk, messagebox

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

try:
    import zstandard as zstd
except Exception:
    zstd = None

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    Figure = None
    FigureCanvasTkAgg = None


# ============================================================
# CDR READER
# ============================================================

class CDRReader:
    """
    Very small CDR decoder for common ROS2 primitive messages.

    ROS2 serialized data usually starts with 4 bytes:
        00 01 00 00 for little endian CDR
    After that, fields are aligned.
    """
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.endian = "<"

        if len(data) >= 4:
            # CDR encapsulation
            # 0x0001 commonly means little-endian CDR.
            enc = data[:2]
            if enc in (b"\x00\x01", b"\x01\x00"):
                self.pos = 4
                self.endian = "<"

    def align(self, n: int):
        pad = (n - (self.pos % n)) % n
        self.pos += pad

    def read_i32(self) -> int:
        self.align(4)
        v = struct.unpack_from(self.endian + "i", self.data, self.pos)[0]
        self.pos += 4
        return v

    def read_u32(self) -> int:
        self.align(4)
        v = struct.unpack_from(self.endian + "I", self.data, self.pos)[0]
        self.pos += 4
        return v

    def read_f32(self) -> float:
        self.align(4)
        v = struct.unpack_from(self.endian + "f", self.data, self.pos)[0]
        self.pos += 4
        return v

    def read_f64(self) -> float:
        self.align(8)
        v = struct.unpack_from(self.endian + "d", self.data, self.pos)[0]
        self.pos += 8
        return v

    def read_string(self) -> str:
        self.align(4)
        n = self.read_u32()
        if n <= 0:
            return ""
        raw = self.data[self.pos:self.pos + n]
        self.pos += n
        if raw.endswith(b"\x00"):
            raw = raw[:-1]
        return raw.decode("utf-8", errors="replace")

    def read_time(self) -> Dict[str, int]:
        sec = self.read_i32()
        nanosec = self.read_u32()
        return {"sec": sec, "nanosec": nanosec}

    def read_header(self) -> Dict[str, Any]:
        stamp = self.read_time()
        frame_id = self.read_string()
        return {"stamp": stamp, "frame_id": frame_id}

    def read_vector3(self) -> Dict[str, float]:
        return {
            "x": self.read_f64(),
            "y": self.read_f64(),
            "z": self.read_f64(),
        }


def _read_multiarray_layout(reader: CDRReader) -> Dict[str, Any]:
    """
    Decode std_msgs/MultiArrayLayout enough to skip it cleanly.
    layout:
      MultiArrayDimension[] dim
      uint32 data_offset
    dimension:
      string label
      uint32 size
      uint32 stride
    """
    dims = []
    dim_len = reader.read_u32()
    for _ in range(dim_len):
        dims.append({
            "label": reader.read_string(),
            "size": reader.read_u32(),
            "stride": reader.read_u32(),
        })
    data_offset = reader.read_u32()
    return {"dim": dims, "data_offset": data_offset}


def decode_msg(msg_type: str, data: bytes) -> Any:
    r = CDRReader(data)

    try:
        if msg_type == "std_msgs/msg/Int32":
            return r.read_i32()

        if msg_type == "std_msgs/msg/Float32":
            return round(r.read_f32(), 6)

        if msg_type == "std_msgs/msg/Float64":
            return round(r.read_f64(), 6)

        if msg_type == "std_msgs/msg/String":
            return r.read_string()

        if msg_type == "std_msgs/msg/Int32MultiArray":
            layout = _read_multiarray_layout(r)
            n = r.read_u32()
            values = [r.read_i32() for _ in range(n)]
            return {"layout": layout, "data": values}

        if msg_type == "std_msgs/msg/Float32MultiArray":
            layout = _read_multiarray_layout(r)
            n = r.read_u32()
            values = [round(r.read_f32(), 6) for _ in range(n)]
            return {"layout": layout, "data": values}

        if msg_type == "std_msgs/msg/Float64MultiArray":
            layout = _read_multiarray_layout(r)
            n = r.read_u32()
            values = [round(r.read_f64(), 6) for _ in range(n)]
            return {"layout": layout, "data": values}

        if msg_type == "geometry_msgs/msg/Vector3":
            return r.read_vector3()

        if msg_type == "geometry_msgs/msg/Twist":
            return {
                "linear": r.read_vector3(),
                "angular": r.read_vector3(),
            }

        if msg_type == "geometry_msgs/msg/TwistStamped":
            return {
                "header": r.read_header(),
                "twist": {
                    "linear": r.read_vector3(),
                    "angular": r.read_vector3(),
                }
            }

    except Exception as e:
        return {
            "decode_error": str(e),
            "raw_hex": data[:80].hex(" ")
        }

    return {
        "unsupported_type": msg_type,
        "raw_hex": data[:80].hex(" ")
    }


def flatten_numeric(value: Any, prefix: str = "") -> Dict[str, float]:
    """
    Extract numbers from decoded messages for plotting.
    """
    out = {}

    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            out[prefix or "value"] = float(value)
        return out

    if isinstance(value, list):
        for i, v in enumerate(value):
            out.update(flatten_numeric(v, f"{prefix}[{i}]"))
        return out

    if isinstance(value, dict):
        # For MultiArray, plot data values directly.
        if "data" in value and isinstance(value["data"], list):
            for i, v in enumerate(value["data"]):
                if isinstance(v, (int, float)):
                    out[f"data[{i}]"] = float(v)

        for k, v in value.items():
            if k in ("layout", "header", "stamp", "frame_id"):
                continue
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_numeric(v, new_prefix))

    return out


def pretty(value: Any, indent: int = 0) -> str:
    sp = " " * indent

    if isinstance(value, dict):
        lines = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}{k}:")
                lines.append(pretty(v, indent + 2))
            else:
                lines.append(f"{sp}{k}: {v}")
        return "\n".join(lines)

    if isinstance(value, list):
        if len(value) <= 20:
            return sp + str(value)
        return sp + str(value[:20]) + f" ... ({len(value)} values)"

    return sp + str(value)


# ============================================================
# BAG LOADING
# ============================================================

@dataclass
class TopicInfo:
    topic_id: int
    name: str
    msg_type: str
    count: int = 0


@dataclass
class Msg:
    t_ns: int
    topic_id: int
    data: bytes


class BagData:
    def __init__(self):
        self.path: Optional[Path] = None
        self.db_path: Optional[Path] = None
        self.topics: Dict[int, TopicInfo] = {}
        self.messages: List[Msg] = []
        self.by_topic: Dict[int, List[Msg]] = {}
        self.t0_ns: int = 0
        self.t1_ns: int = 0

    @property
    def duration_s(self) -> float:
        if not self.messages:
            return 0.0
        return (self.t1_ns - self.t0_ns) / 1e9

    def time_s(self, ns: int) -> float:
        return (ns - self.t0_ns) / 1e9


def find_db3(path: Path) -> Path:
    if path.is_file():
        return path

    candidates = list(path.glob("*.db3")) + list(path.glob("*.db3.zstd")) + list(path.glob("*.sqlite3"))
    if not candidates:
        candidates = list(path.rglob("*.db3")) + list(path.rglob("*.db3.zstd"))

    if not candidates:
        raise FileNotFoundError("Aucun fichier .db3 ou .db3.zstd trouvé dans ce dossier.")

    return candidates[0]


def maybe_decompress_zstd(db_path: Path) -> Path:
    if db_path.suffix != ".zstd":
        return db_path

    if zstd is None:
        raise RuntimeError(
            "Le bag est compressé en .zstd. Installe zstandard:\n\n"
            "    pip install zstandard\n"
        )

    out_dir = Path(tempfile.mkdtemp(prefix="rosbag_viewer_"))
    out_path = out_dir / db_path.name.replace(".zstd", "")

    dctx = zstd.ZstdDecompressor()
    with open(db_path, "rb") as f_in, open(out_path, "wb") as f_out:
        dctx.copy_stream(f_in, f_out)

    return out_path


def load_bag(path: Path) -> BagData:
    bag = BagData()
    bag.path = path

    db = find_db3(path)
    db = maybe_decompress_zstd(db)
    bag.db_path = db

    conn = sqlite3.connect(str(db))
    cur = conn.cursor()

    # topics table generally:
    # id, name, type, serialization_format, offered_qos_profiles
    for row in cur.execute("SELECT id, name, type FROM topics ORDER BY id"):
        topic_id, name, msg_type = row
        bag.topics[topic_id] = TopicInfo(topic_id, name, msg_type)

    # messages table:
    # id, topic_id, timestamp, data
    rows = cur.execute(
        "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp"
    ).fetchall()

    for topic_id, timestamp, data in rows:
        m = Msg(int(timestamp), int(topic_id), bytes(data))
        bag.messages.append(m)
        bag.by_topic.setdefault(int(topic_id), []).append(m)
        if int(topic_id) in bag.topics:
            bag.topics[int(topic_id)].count += 1

    conn.close()

    if bag.messages:
        bag.t0_ns = bag.messages[0].t_ns
        bag.t1_ns = bag.messages[-1].t_ns

    return bag


def find_latest_before(msgs: List[Msg], target_ns: int) -> Optional[Msg]:
    if not msgs:
        return None

    lo, hi = 0, len(msgs) - 1
    best = None

    while lo <= hi:
        mid = (lo + hi) // 2
        if msgs[mid].t_ns <= target_ns:
            best = msgs[mid]
            lo = mid + 1
        else:
            hi = mid - 1

    return best


# ============================================================
# GUI
# ============================================================

class RosbagViewer(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Mini ROS2 Bag Viewer - Select bag/images - No ROS2 needed")
        self.geometry("1200x760")

        self.bag: Optional[BagData] = None
        self.selected_topic_id: Optional[int] = None

        self.image_files: List[Path] = []
        self.current_photo = None

        self.bag_path_var = tk.StringVar()
        self.image_dir_var = tk.StringVar()
        self.annotated_fps_var = tk.StringVar(value="10.0")

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(top, text="Rosbag:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.bag_entry = ttk.Entry(top, textvariable=self.bag_path_var, width=70)
        self.bag_entry.grid(row=0, column=1, sticky="ew", padx=4)

        ttk.Button(top, text="Browse bag", command=self.browse_bag).grid(row=0, column=2, padx=4)

        ttk.Label(top, text="Images:").grid(row=1, column=0, sticky="w", padx=(0, 4), pady=(4, 0))
        self.image_entry = ttk.Entry(top, textvariable=self.image_dir_var, width=70)
        self.image_entry.grid(row=1, column=1, sticky="ew", padx=4, pady=(4, 0))

        ttk.Button(top, text="Browse images", command=self.browse_images).grid(row=1, column=2, padx=4, pady=(4, 0))

        ttk.Label(top, text="Image FPS:").grid(row=0, column=3, sticky="e", padx=(12, 4))
        ttk.Entry(top, textvariable=self.annotated_fps_var, width=8).grid(row=0, column=4, sticky="w")

        ttk.Button(top, text="Load", command=self.load_selected_paths).grid(row=1, column=3, columnspan=2, sticky="ew", padx=(12, 0), pady=(4, 0))

        top.columnconfigure(1, weight=1)

        self.path_label = ttk.Label(self, text="No bag loaded")
        self.path_label.pack(fill=tk.X, padx=8, pady=(0, 4))

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(main, width=390)
        right = ttk.Frame(main)
        main.add(left, weight=0)
        main.add(right, weight=1)

        ttk.Label(left, text="Topics").pack(anchor=tk.W)

        cols = ("name", "type", "count")
        self.topic_tree = ttk.Treeview(left, columns=cols, show="headings", height=24)
        self.topic_tree.heading("name", text="Topic")
        self.topic_tree.heading("type", text="Type")
        self.topic_tree.heading("count", text="Count")
        self.topic_tree.column("name", width=210)
        self.topic_tree.column("type", width=140)
        self.topic_tree.column("count", width=60, anchor=tk.E)
        self.topic_tree.pack(fill=tk.BOTH, expand=True, pady=4)
        self.topic_tree.bind("<<TreeviewSelect>>", self.on_topic_select)

        time_frame = ttk.Frame(right)
        time_frame.pack(fill=tk.X)

        self.time_label = ttk.Label(time_frame, text="Time: 0.000 s")
        self.time_label.pack(anchor=tk.W)

        self.slider = ttk.Scale(time_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_slider)
        self.slider.pack(fill=tk.X, pady=4)

        self.msg_info_label = ttk.Label(right, text="Selected message: -")
        self.msg_info_label.pack(anchor=tk.W, pady=(8, 2))

        split = ttk.PanedWindow(right, orient=tk.VERTICAL)
        split.pack(fill=tk.BOTH, expand=True)

        text_frame = ttk.Frame(split)
        bottom_split = ttk.PanedWindow(split, orient=tk.HORIZONTAL)
        image_frame = ttk.Frame(bottom_split)
        plot_frame = ttk.Frame(bottom_split)

        split.add(text_frame, weight=1)
        split.add(bottom_split, weight=2)
        bottom_split.add(image_frame, weight=1)
        bottom_split.add(plot_frame, weight=1)

        ttk.Label(text_frame, text="Decoded latest message before slider time").pack(anchor=tk.W)

        self.text = tk.Text(text_frame, wrap=tk.NONE, height=16)
        self.text.pack(fill=tk.BOTH, expand=True)

        xscroll = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL, command=self.text.xview)
        xscroll.pack(fill=tk.X)
        self.text.configure(xscrollcommand=xscroll.set)

        ttk.Label(image_frame, text="Annotated image near slider time").pack(anchor=tk.W)
        self.image_label = ttk.Label(image_frame, text="No annotated images found", anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        if Figure is not None:
            self.fig = Figure(figsize=(6, 3), dpi=100)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlabel("time [s]")
            self.ax.set_ylabel("value")
            self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            self.fig = None
            self.ax = None
            self.canvas = None
            ttk.Label(plot_frame, text="Install matplotlib to enable plots:\n\npip install matplotlib").pack(pady=20)

    def browse_bag(self):
        folder = filedialog.askdirectory(title="Select rosbag folder")
        if folder:
            self.bag_path_var.set(folder)

            img_dir = Path(folder) / "last_annotated_frames"
            if img_dir.exists() and img_dir.is_dir():
                self.image_dir_var.set(str(img_dir))
            return

        file_path = filedialog.askopenfilename(
            title="Or select .db3/.db3.zstd file",
            filetypes=[("ROS bag db", "*.db3 *.db3.zstd *.sqlite3"), ("All files", "*.*")]
        )
        if file_path:
            self.bag_path_var.set(file_path)

            img_dir = Path(file_path).parent / "last_annotated_frames"
            if img_dir.exists() and img_dir.is_dir():
                self.image_dir_var.set(str(img_dir))


    def browse_images(self):
        folder = filedialog.askdirectory(title="Select annotated images folder")
        if folder:
            self.image_dir_var.set(folder)
            self.load_images_only()


    def load_selected_paths(self):
        bag_path = self.bag_path_var.get().strip()

        if not bag_path:
            messagebox.showerror("Missing path", "Select a rosbag folder or .db3/.db3.zstd file first.")
            return

        try:
            self.bag = load_bag(Path(bag_path))
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return

        self.load_images_only(show_errors=False)

        self.path_label.config(
            text=(
                f"{self.bag.path} | {len(self.bag.messages)} messages | "
                f"{self.bag.duration_s:.2f}s | {len(self.image_files)} annotated images"
            )
        )

        self.topic_tree.delete(*self.topic_tree.get_children())

        for topic_id, info in sorted(self.bag.topics.items(), key=lambda x: x[1].name):
            self.topic_tree.insert(
                "",
                tk.END,
                iid=str(topic_id),
                values=(info.name, info.msg_type, info.count)
            )

        self.slider.config(from_=0, to=max(self.bag.duration_s, 0.001))
        self.slider.set(0)

        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, "Bag loaded. Select a topic on the left.\n")
        self.update_image()


    def load_images_only(self, show_errors=True):
        image_dir = self.image_dir_var.get().strip()

        if not image_dir:
            self.image_files = []
            self.update_image()
            return

        folder = Path(image_dir)

        if not folder.exists() or not folder.is_dir():
            self.image_files = []
            if show_errors:
                messagebox.showerror("Image folder error", f"Image folder does not exist:\n{folder}")
            self.update_image()
            return

        self.image_files = self.load_images_from_folder(folder)

        if self.bag is not None:
            self.path_label.config(
                text=(
                    f"{self.bag.path} | {len(self.bag.messages)} messages | "
                    f"{self.bag.duration_s:.2f}s | {len(self.image_files)} annotated images"
                )
            )

        self.update_image()


    def on_topic_select(self, _event=None):
        sel = self.topic_tree.selection()
        if not sel or self.bag is None:
            return

        self.selected_topic_id = int(sel[0])
        self.update_current_message()
        self.update_plot()

    def on_slider(self, _value=None):
        if self.bag is None:
            return

        t = float(self.slider.get())
        self.time_label.config(text=f"Time: {t:.3f} s / {self.bag.duration_s:.3f} s")
        self.update_current_message()
        self.update_image()

    def update_current_message(self):
        if self.bag is None or self.selected_topic_id is None:
            return

        target_ns = self.bag.t0_ns + int(float(self.slider.get()) * 1e9)
        msgs = self.bag.by_topic.get(self.selected_topic_id, [])
        msg = find_latest_before(msgs, target_ns)

        self.text.delete("1.0", tk.END)

        info = self.bag.topics[self.selected_topic_id]

        if msg is None:
            self.msg_info_label.config(text="Selected message: none before this time")
            self.text.insert(tk.END, "No message before current slider time.")
            return

        decoded = decode_msg(info.msg_type, msg.data)

        self.msg_info_label.config(
            text=(
                f"Selected message: topic={info.name} | "
                f"type={info.msg_type} | "
                f"t={self.bag.time_s(msg.t_ns):.3f}s"
            )
        )

        self.text.insert(tk.END, pretty(decoded))

    def load_images_from_folder(self, folder: Path) -> List[Path]:
        """
        Load images directly from the folder specified by the user.
        Usually:
            mission_debug_xxx/last_annotated_frames/

        The images are assumed to be the last N frames before the rosbag stopped.
        Example:
            annotated_000.jpg = oldest of the last frames
            annotated_099.jpg = latest frame before stop
        """
        images = []

        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            images.extend(folder.glob(ext))

        def sort_key(p: Path):
            nums = re.findall(r"\d+", p.stem)
            return int(nums[-1]) if nums else p.name

        return sorted(set(images), key=sort_key)


    def image_for_time(self, t_s: float) -> Optional[Path]:
        if not self.image_files or self.bag is None:
            return None

        if len(self.image_files) == 1:
            return self.image_files[0]

        duration = self.bag.duration_s

        try:
            fps = float(self.annotated_fps_var.get())
            if fps <= 0:
                fps = 10.0
        except Exception:
            fps = 10.0

        image_window_s = min(duration, len(self.image_files) / fps)
        start_s = duration - image_window_s

        if t_s < start_s:
            return None

        ratio = (t_s - start_s) / image_window_s
        ratio = max(0.0, min(1.0, ratio))

        idx = round(ratio * (len(self.image_files) - 1))
        return self.image_files[idx]


    def update_image(self):
        if not hasattr(self, "image_label"):
            return

        if Image is None or ImageTk is None:
            self.image_label.config(text="Install Pillow to show images:\n\npip install pillow", image="")
            return

        if self.bag is None or not self.image_files:
            self.image_label.config(text="No annotated images found", image="")
            return

        t = float(self.slider.get())
        img_path = self.image_for_time(t)

        if img_path is None:
            self.image_label.config(
                text="No annotated image yet\n(images are only the last 100 frames)",
                image=""
            )
            return

        try:
            img = Image.open(img_path).convert("RGB")

            max_w = max(self.image_label.winfo_width(), 400)
            max_h = max(self.image_label.winfo_height(), 300)

            img.thumbnail((max_w, max_h))

            self.current_photo = ImageTk.PhotoImage(img)
            self.image_label.config(
                image=self.current_photo,
                text=f"{img_path.name}",
                compound=tk.TOP
            )

        except Exception as e:
            self.image_label.config(text=f"Could not load image:\n{img_path}\n\n{e}", image="")


    def update_plot(self):
        if (
            self.bag is None
            or self.selected_topic_id is None
            or self.ax is None
            or self.canvas is None
        ):
            return

        info = self.bag.topics[self.selected_topic_id]
        msgs = self.bag.by_topic.get(self.selected_topic_id, [])

        series: Dict[str, Tuple[List[float], List[float]]] = {}

        for msg in msgs:
            decoded = decode_msg(info.msg_type, msg.data)
            nums = flatten_numeric(decoded)
            t = self.bag.time_s(msg.t_ns)

            for name, val in nums.items():
                xs, ys = series.setdefault(name, ([], []))
                xs.append(t)
                ys.append(val)

        self.ax.clear()
        self.ax.set_xlabel("time [s]")
        self.ax.set_ylabel("value")

        if not series:
            self.ax.text(0.5, 0.5, "No numeric values to plot", ha="center", va="center")
        else:
            for name, (xs, ys) in list(series.items())[:8]:
                self.ax.plot(xs, ys, label=name)
            self.ax.legend(loc="best")

        self.canvas.draw()


if __name__ == "__main__":
    app = RosbagViewer()
    app.mainloop()
