import cv2
import depthai as dai
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ultralytics import YOLO
# from scripts.Point_milieu import *
from scripts.Depth_and_Angle.Detection_Orange import detect_orange_boxes, draw_orange_boxes


# ---------------- CONFIG ----------------
USE_CAMERA = True  # True = camera, False = folder
SCRIPT_DIR = Path(__file__).parent.resolve()
IMAGE_DIR = SCRIPT_DIR / "datasets/Test_Piscine_a_annoter/Tests_march_18/rgb"
LABEL_DIR = SCRIPT_DIR / "datasets/Test_Piscine_a_annoter/Tests_march_18/labels"
IMAGE_EXT = ".png"
NUM_IMAGES = 10
MAX_FRAMES = 100
PAUSE_KEY = ord('p')
ESC_KEY = 27
PREDICT_MODE = False
MODEL_PATH = SCRIPT_DIR / "datasets/Tests_Datasets_Roboflow/Data_1/runs/detect/train3/weights/best.pt"
# MODEL_PATH_IMG = SCRIPT_DIR / "datasets/Tests_Datasets_Roboflow/Data_1/valid/images"
MODEL_PATH_IMG = SCRIPT_DIR / "datasets/Test_Piscine_a_annoter/Tests_march_18/rgb"
frames = []


# ---------------- INPUT SOURCE ----------------
if USE_CAMERA:

    pipeline = dai.Pipeline()

    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    stereo = pipeline.create(dai.node.StereoDepth)

    rgbOut = rgb.requestOutput(
        size=(1280, 720),
        fps=15,
        type=dai.ImgFrame.Type.RGB888i,
    )

    monoLeftOut = monoLeft.requestFullResolutionOutput()
    monoRightOut = monoRight.requestFullResolutionOutput()
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

    monoQueue = monoLeftOut.createOutputQueue(maxSize=4)
    monoQueueR = monoRightOut.createOutputQueue(maxSize=4)
    rgbQueue = rgbOut.createOutputQueue(maxSize=4)
    dispQueue = stereo.disparity.createOutputQueue()
    depthQueue = stereo.depth.createOutputQueue(maxSize=4)

    inDepth = None

    # -------------- Example usage --------------
    with pipeline:
        pipeline.start()

        while pipeline.isRunning():
            inRgb = rgbQueue.get()
            frame = inRgb.getCvFrame()

            inDepth = depthQueue.get()
            if inDepth is not None:
                depth_frame = inDepth.getFrame()

            # Si nécessaire, convertis en BGR pour affichage OpenCV
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Détection orange
            boxes, mask = detect_orange_boxes(frame, depth_frame, min_area=900)
            vis = draw_orange_boxes(frame, boxes)

            # Affichage
            cv2.imshow("rgb_orange_detection", vis)
            cv2.imshow("orange_mask", mask)


            key = cv2.waitKey(1)
            if key == PAUSE_KEY:
                cv2.waitKey(0)
            if key == ESC_KEY:
                pipeline.stop()
                break
# --------------------------------------------

elif PREDICT_MODE:
    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=str(MODEL_PATH_IMG),
        save=True,
        conf=0.10
    )

else:
    print(f"Loading {NUM_IMAGES} images from {IMAGE_DIR}...")

    image_paths = sorted(IMAGE_DIR.glob(f"*{IMAGE_EXT}"))

    for img_path in image_paths[:NUM_IMAGES]:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Warning: Could not load {img_path}")
            continue

        frames.append(frame)

    # -------------- Example usage --------------
    for i, frame in enumerate(frames):
        print(f"Frame {i} shape: {frame.shape}")

        boxes, mask = detect_orange_boxes(frame, min_area=900)
        vis = draw_orange_boxes(frame, boxes)

        cv2.imshow("rgb_orange_detection", vis)
        cv2.imshow("orange_mask", mask)
        cv2.waitKey(0)


cv2.destroyAllWindows()
