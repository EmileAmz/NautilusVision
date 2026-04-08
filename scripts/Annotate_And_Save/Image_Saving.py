import depthai as dai
import cv2
import numpy as np
import time
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
documents = Path.home() / "Documents"
dataset_folder = documents / "dataset"

RGB_FOLDER = dataset_folder / "rgb"
DEPTH_FOLDER = dataset_folder / "depth"

RGB_FOLDER.mkdir(parents=True, exist_ok=True)
DEPTH_FOLDER.mkdir(parents=True, exist_ok=True)

FPS = 10
PERIODIC_INTERVAL = 1.0
BURST_COUNT = 5
BURST_INTERVAL = 0.5
DEPTH_EVENT_THRESHOLD = 200  # mm
CAM = "CAM_DESSOUS"  # "CAM_DESSOUS" ou "CAM_AVANT"

# =========================
# PIPELINE CREATION
# =========================
pipeline = dai.Pipeline()

# RGB camera
camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
cam_q_in = camRgb.inputControl.createInputQueue()

camRgbOut = camRgb.requestOutput(
    size=(1280, 720),
    fps=FPS,
    type=dai.ImgFrame.Type.NV12,
)

depthQueue = None

if CAM == "CAM_AVANT":
    # Mono cameras
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    monoLeftOut = monoLeft.requestOutput(size=(1280, 720), fps=FPS)
    monoRightOut = monoRight.requestOutput(size=(1280, 720), fps=FPS)

    # Stereo depth
    stereo = pipeline.create(dai.node.StereoDepth)
    monoLeftOut.link(stereo.left)
    monoRightOut.link(stereo.right)

    stereo.setRectification(True)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setExtendedDisparity(False)

    # Align depth to RGB camera
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    # Width must be multiple of 16
    stereo.setOutputSize(1280, 720)

    # Depth queue
    depthQueue = stereo.depth.createOutputQueue(maxSize=4)

# RGB queue
rgbQueue = camRgbOut.createOutputQueue(maxSize=4)

# =========================
# HELPER FUNCTIONS
# =========================
def depth_to_colormap(depth_frame, max_depth_mm=5000):
    depth = depth_frame.astype(np.float32).copy()
    depth[depth <= 0] = np.nan
    depth = np.clip(depth, 0, max_depth_mm)

    depth_vis = np.nan_to_num(depth, nan=0.0)
    depth_vis = (depth_vis / max_depth_mm) * 255.0
    depth_vis = np.clip(depth_vis, 0, 255).astype(np.uint8)

    return cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)

# =========================
# RUN DEVICE
# =========================
with pipeline:
    pipeline.start()

    mode = 2
    last_capture_time = 0
    last_depth_mean = None

    print("\n--- MODES ---")
    print("1: Periodic")
    print("2: Manual (SPACE)")
    print("3: Event (Depth change) [CAM_AVANT seulement]")
    print("4: Burst (SPACE)")
    print("5: Pause")
    print("d: Print depth info [CAM_AVANT seulement]")
    print("ESC: Quit\n")

    ctrl = dai.CameraControl()
    ctrl.setManualExposure(20000, 400)  # 20 ms, ISO 400
    cam_q_in.send(ctrl)

    if CAM == "CAM_DESSOUS":
        ctrl.setManualFocus(120)


    while pipeline.isRunning():
        inRgb = rgbQueue.get()
        if inRgb is None:
            continue

        rgb_frame = inRgb.getCvFrame()  # BGR for OpenCV
        depth_frame = None
        center_depth = None

        if CAM == "CAM_AVANT" and depthQueue is not None:
            inDepth = depthQueue.get()
            if inDepth is not None:
                depth_frame = inDepth.getFrame()

                # Depth visualization
                depth_vis = depth_to_colormap(depth_frame, max_depth_mm=5000)

                # Center pixel debug
                h, w = rgb_frame.shape[:2]
                cx = w // 2
                cy = h // 2

                center_patch = depth_frame[
                    max(0, cy - 5):min(h, cy + 5),
                    max(0, cx - 5):min(w, cx + 5)
                ]
                valid_center = center_patch[(center_patch > 200) & (center_patch < 5000)]

                if valid_center.size > 0:
                    center_depth = float(np.median(valid_center))
                    depth_text = f"Center depth: {center_depth:.0f} mm"
                else:
                    depth_text = "Center depth: invalid"

                cv2.circle(rgb_frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.circle(depth_vis, (cx, cy), 4, (0, 255, 0), -1)

                cv2.putText(
                    rgb_frame,
                    depth_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.imshow("Depth aligned to RGB", depth_vis)

        cv2.imshow("RGB", rgb_frame)

        key = cv2.waitKey(1) & 0xFF
        timestamp = time.time()

        # =========================
        # MODE SWITCH
        # =========================
        if key == ord('1'):
            mode = 1
            print("Mode: Periodic")

        elif key == ord('2'):
            mode = 2
            print("Mode: Manual")

        elif key == ord('3'):
            if CAM == "CAM_AVANT":
                mode = 3
                print("Mode: Event-based")
            else:
                print("Mode 3 indisponible avec CAM_DESSOUS")

        elif key == ord('4'):
            mode = 4
            print("Mode: Burst")

        elif key == ord('5'):
            mode = 5
            print("Mode: Pause")

        elif key == ord('d'):
            if depth_frame is not None:
                print("dtype:", depth_frame.dtype)
                print("min:", np.min(depth_frame))
                print("max:", np.max(depth_frame))
                if center_depth is not None:
                    print(f"center depth: {center_depth:.1f} mm")
                else:
                    print("center depth: invalid")
            else:
                print("Pas de depth disponible pour cette caméra")

        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            break

        filename = f"{timestamp:.3f}.png"

        # =========================
        # MODE 1: PERIODIC
        # =========================
        if mode == 1:
            if timestamp - last_capture_time >= PERIODIC_INTERVAL:
                cv2.imwrite(str(RGB_FOLDER / filename), rgb_frame)

                if depth_frame is not None:
                    cv2.imwrite(str(DEPTH_FOLDER / filename), depth_frame)

                print("Saved periodic:", filename)
                last_capture_time = timestamp

        # =========================
        # MODE 2: MANUAL
        # =========================
        elif mode == 2:
            if key == 32:  # SPACE
                cv2.imwrite(str(RGB_FOLDER / filename), rgb_frame)

                if depth_frame is not None:
                    cv2.imwrite(str(DEPTH_FOLDER / filename), depth_frame)

                print("Saved manual:", filename)

        # =========================
        # MODE 3: EVENT BASED
        # =========================
        elif mode == 3:
            if depth_frame is not None:
                valid_depth = depth_frame[(depth_frame > 200) & (depth_frame < 5000)]

                if valid_depth.size > 0:
                    depth_mean = float(np.median(valid_depth))

                    if last_depth_mean is not None:
                        if abs(depth_mean - last_depth_mean) > DEPTH_EVENT_THRESHOLD:
                            cv2.imwrite(str(RGB_FOLDER / filename), rgb_frame)
                            cv2.imwrite(str(DEPTH_FOLDER / filename), depth_frame)
                            print("Saved event:", filename)

                    last_depth_mean = depth_mean

        # =========================
        # MODE 4: BURST
        # =========================
        elif mode == 4:
            if key == 32:  # SPACE
                print("Burst started")
                for i in range(BURST_COUNT):
                    burst_time = time.time()
                    burst_filename = f"{burst_time:.3f}.png"

                    cv2.imwrite(str(RGB_FOLDER / burst_filename), rgb_frame)

                    if depth_frame is not None:
                        cv2.imwrite(str(DEPTH_FOLDER / burst_filename), depth_frame)

                    print("Saved burst:", burst_filename)
                    time.sleep(BURST_INTERVAL)

cv2.destroyAllWindows()