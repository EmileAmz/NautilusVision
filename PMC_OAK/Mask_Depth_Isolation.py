import cv2
import depthai as dai
import numpy as np

# ===================== CONFIG =====================
MIN_DISP = 1000       # start conservative
MAX_DISP = 6000     # normal indoors with extended disparity

MIN_AREA_RATIO = 0.01
MAX_AREA_RATIO = 0.4

ASPECT_MIN = 0.2
ASPECT_MAX = 5.0

SMOOTH_ALPHA = 0.7
MAX_MISSES = 5
ROI_SIZE = 160

DEPTH_STD_MIN = 1.5     # reject flat surfaces (walls)
FULL_FRAME_RATIO = 0.8 # reject wall-sized blobs

HIST_MAX = 5000     # adjust if you see higher
HIST_BINS = 10

ESC_KEY = 27
# ==================================================

# ---------- Simple temporal tracker ----------
class TrackedROI:
    def __init__(self, bbox):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.misses = 0

    def update(self, bbox):
        self.bbox = (
            SMOOTH_ALPHA * self.bbox +
            (1 - SMOOTH_ALPHA) * np.array(bbox)
        )
        self.misses = 0

tracked = []

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax+aw, bx+bw)
    iy2 = min(ay+ah, by+bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    inter = (ix2-ix1)*(iy2-iy1)
    union = aw*ah + bw*bh - inter
    return inter / union

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        disp = param
        val = disp[y, x]
        print(f"Disparity at ({x},{y}) = {val}")

def draw_disparity_histogram(disp, hist_max=2000, bins=100):
    valid = disp[disp > 0]
    h, w = 240, 420

    hist_img = np.zeros((h, w, 3), dtype=np.uint8)

    if len(valid) == 0:
        cv2.putText(hist_img, "No valid disparity",
                    (80, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        return hist_img

    # Histogram
    hist, _ = np.histogram(valid, bins=bins, range=(0, hist_max))
    hist = hist.astype(np.float32)
    hist /= hist.max()

    plot_h = 160
    plot_w = w - 40
    offset_x = 20
    offset_y = 30

    bin_width = plot_w / bins

    # Draw bars
    for i in range(bins):
        x1 = int(offset_x + i * bin_width)
        x2 = int(offset_x + (i + 1) * bin_width)
        bar_h = int(hist[i] * plot_h)

        cv2.rectangle(
            hist_img,
            (x1, offset_y + plot_h),
            (x2, offset_y + plot_h - bar_h),
            (255, 255, 255),
            -1
        )

    # ---- LEGEND / AXIS ----
    tick_values = [0, 500, 1000, 1500, 2000]

    for v in tick_values:
        x = int(offset_x + (v / hist_max) * plot_w)

        # Vertical tick line
        cv2.line(hist_img,
                 (x, offset_y),
                 (x, offset_y + plot_h),
                 (80, 80, 80), 1)

        # Numeric label
        cv2.putText(hist_img, str(v),
                    (x - 12, offset_y + plot_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (200, 200, 200), 1)

    # Axis box
    cv2.rectangle(hist_img,
                  (offset_x, offset_y),
                  (offset_x + plot_w, offset_y + plot_h),
                  (150, 150, 150), 1)

    # Title
    cv2.putText(hist_img, "Raw Disparity Histogram",
                (70, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2)

    return hist_img



cv2.namedWindow("raw disparity")

# ---------- DepthAI pipeline ----------
pipeline = dai.Pipeline()

monoL = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoR = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
stereo = pipeline.create(dai.node.StereoDepth)

monoL.requestFullResolutionOutput().link(stereo.left)
monoR.requestFullResolutionOutput().link(stereo.right)

stereo.setRectification(True)
stereo.setExtendedDisparity(True)
stereo.setLeftRightCheck(True)

dispQ = stereo.disparity.createOutputQueue()
monoQ = monoL.requestFullResolutionOutput().createOutputQueue()

# ---------- Run ----------
with pipeline:
    pipeline.start()

    while pipeline.isRunning():

        mono = monoQ.get().getFrame()
        disp = dispQ.get().getFrame()

        hist_img = draw_disparity_histogram(disp, hist_max=HIST_MAX, bins=HIST_BINS)
        cv2.imshow("disparity histogram", hist_img)

        H, W = mono.shape
        debug = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

        # ---------- RAW disparity visualization (DEBUG) ----------
        disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = disp_vis.astype(np.uint8)
        cv2.setMouseCallback("raw disparity", on_mouse, disp)

        cv2.imshow("raw disparity", disp_vis)

        # ---------- Depth mask ----------
        depthMask = (disp > MIN_DISP) & (disp < MAX_DISP)
        depthMask = depthMask.astype(np.uint8) * 255

        kernel = np.ones((5,5), np.uint8)
        depthMask = cv2.morphologyEx(depthMask, cv2.MORPH_OPEN, kernel)
        depthMask = cv2.morphologyEx(depthMask, cv2.MORPH_CLOSE, kernel)

        # ---------- Connected components ----------
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            depthMask, connectivity=8
        )

        imgArea = H * W
        detections = []

        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]

            if area < MIN_AREA_RATIO * imgArea:
                continue
            if area > MAX_AREA_RATIO * imgArea:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # ---------- Reject wall-sized blobs ----------
            if w > FULL_FRAME_RATIO * W or h > FULL_FRAME_RATIO * H:
                continue

            aspect = w / h if h > 0 else 0
            if aspect < ASPECT_MIN or aspect > ASPECT_MAX:
                continue

            # ---------- Depth variance filter ----------
            roi_disp = disp[y:y+h, x:x+w]
            valid = roi_disp[roi_disp > 0]
            if len(valid) < 50:
                continue
            if np.std(valid) < DEPTH_STD_MIN:
                continue

            detections.append((x,y,w,h))

        # ---------- Temporal association ----------
        updated = [False]*len(tracked)

        for det in detections:
            best_iou = 0
            best_idx = -1
            for i, trk in enumerate(tracked):
                score = iou(trk.bbox, det)
                if score > best_iou:
                    best_iou = score
                    best_idx = i

            if best_iou > 0.3:
                tracked[best_idx].update(det)
                updated[best_idx] = True
            else:
                tracked.append(TrackedROI(det))
                updated.append(True)

        for i, trk in enumerate(tracked):
            if not updated[i]:
                trk.misses += 1

        tracked[:] = [t for t in tracked if t.misses < MAX_MISSES]

        # ---------- Extract ROIs ----------
        for i, trk in enumerate(tracked):
            x,y,w,h = trk.bbox.astype(int)
            cv2.rectangle(debug, (x,y), (x+w,y+h), (0,255,0), 2)

            roi = mono[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            scale = ROI_SIZE / max(w,h)
            resized = cv2.resize(roi, None, fx=scale, fy=scale)

            padded = np.zeros((ROI_SIZE, ROI_SIZE), dtype=np.uint8)
            padded[:resized.shape[0], :resized.shape[1]] = resized

            cv2.imshow(f"ROI_{i}", padded)

        cv2.imshow("depthMask", depthMask)
        cv2.imshow("debug", debug)

        if cv2.waitKey(1) == ESC_KEY:
            pipeline.stop()
            break
