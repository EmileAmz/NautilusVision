import depthai as dai
import cv2
import numpy as np
import time
import os
from pathlib import Path

# =========================
# CONFIGURATION
# =========================

# Base folder for dataset in Documents
documents = Path.home() / "Documents"
dataset_folder = documents / "dataset"

# RGB / Depth subfolders
RGB_FOLDER = dataset_folder / "rgb"
DEPTH_FOLDER = dataset_folder / "depth"

# Create folders if they don't exist
RGB_FOLDER.mkdir(parents=True, exist_ok=True)
DEPTH_FOLDER.mkdir(parents=True, exist_ok=True)

FPS = 5
PERIODIC_INTERVAL = 1.0        # seconds
BURST_COUNT = 5
BURST_INTERVAL = 0.5           # seconds
DEPTH_EVENT_THRESHOLD = 200    # mm change threshold

# =========================
# PIPELINE CREATION
# =========================

pipeline = dai.Pipeline()

# RGB camera
camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A,)
cam_q_in = camRgb.inputControl.createInputQueue()

camRgbOut = camRgb.requestOutput(
    size=(1280,720),
    fps=FPS,
    type=dai.ImgFrame.Type.NV12,
)

# Mono cameras
monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
stereo = pipeline.create(dai.node.StereoDepth)

monoLeftOut = monoLeft.requestOutput(size=(1280,720),fps=FPS)
monoLeftOut.link(stereo.left)
monoRightOut = monoRight.requestOutput(size=(1280,720),fps=FPS)
monoRightOut.link(stereo.right)

# Stereo depth
stereo.setRectification(True)
stereo.setExtendedDisparity(True)
stereo.setLeftRightCheck(True)

rgbQueue = camRgbOut.createOutputQueue(maxSize=4)
depthQueue = stereo.disparity.createOutputQueue(maxSize=4)

# =========================
# RUN DEVICE
# =========================

with pipeline:
    pipeline.start()
    while pipeline.isRunning():
        mode = 5
        last_capture_time = 0
        last_depth_mean = None

        print("\n--- MODES ---")
        print("1: Periodic")
        print("2: Manual (SPACE)")
        print("3: Event (Depth change)")
        print("4: Burst (SPACE)")
        print("5: Pause")
        print("q: Quit\n")
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(20000, 400)  # 20ms, ISO 400
        # ctrl.setAutoExposureEnable()
        # ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
        cam_q_in.send(ctrl)

        while True:

            inRgb = rgbQueue.get()
            inDepth = depthQueue.get()

            if inRgb is None or inDepth is None:
                continue

            rgb_frame = inRgb.getCvFrame()
            depth_frame = inDepth.getFrame()

            # Depth visualization
            depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = np.uint8(depth_vis)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            cv2.imshow("RGB", rgb_frame)
            cv2.imshow("Depth", depth_vis)

            key = cv2.waitKey(1) & 0xFF

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
                mode = 3
                print("Mode: Event-based")

            elif key == ord('4'):
                mode = 4
                print("Mode: Burst")

            elif key == ord('5'):
                mode = 5
                print("Mode: Pause")

            elif key == 27:
                cv2.destroyAllWindows()
                exit()

            timestamp = time.time()

            # =========================
            # MODE 1: PERIODIC
            # =========================
            if mode == 1:
                if timestamp - last_capture_time >= PERIODIC_INTERVAL:
                    filename = f"{timestamp:.3f}.png"
                    cv2.imwrite(f"{RGB_FOLDER}/{filename}", rgb_frame)
                    cv2.imwrite(f"{DEPTH_FOLDER}/{filename}", depth_frame)
                    print("Saved periodic:", filename)
                    last_capture_time = timestamp

            # =========================
            # MODE 2: MANUAL
            # =========================
            elif mode == 2:
                if key == 32:  # SPACE
                    filename = f"{timestamp:.3f}.png"
                    cv2.imwrite(f"{RGB_FOLDER}/{filename}", rgb_frame)
                    cv2.imwrite(f"{DEPTH_FOLDER}/{filename}", depth_frame)
                    print("Saved manual:", filename)

            # =========================
            # MODE 3: EVENT BASED
            # =========================
            elif mode == 3:
                depth_mean = np.mean(depth_frame[depth_frame > 0])

                if last_depth_mean is not None:
                    if abs(depth_mean - last_depth_mean) > DEPTH_EVENT_THRESHOLD:
                        filename = f"{timestamp:.3f}.png"
                        cv2.imwrite(f"{RGB_FOLDER}/{filename}", rgb_frame)
                        cv2.imwrite(f"{DEPTH_FOLDER}/{filename}", depth_frame)
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
                        filename = f"{burst_time:.3f}.png"
                        cv2.imwrite(f"{RGB_FOLDER}/{filename}", rgb_frame)
                        cv2.imwrite(f"{DEPTH_FOLDER}/{filename}", depth_frame)
                        print("Saved burst:", filename)
                        time.sleep(BURST_INTERVAL)