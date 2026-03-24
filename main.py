import cv2
import depthai as dai
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ---------------- CONFIG ----------------
USE_CAMERA = False  # True = camera, False = folder
SCRIPT_DIR = Path(__file__).parent.resolve()
IMAGE_DIR = SCRIPT_DIR / "datasets/Tests_march_18/dataset/rgb"
LABEL_DIR = SCRIPT_DIR / "datasets/Tests_march_18/dataset/labels"
IMAGE_EXT = ".png"
NUM_IMAGES = 10
MAX_FRAMES = 100
PAUSE_KEY = ord('p')
ESC_KEY = 27
# ---------------------------------------
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
        type=dai.ImgFrame.Type.NV12,
    )
    monoLeftOut = monoLeft.requestFullResolutionOutput()
    monoRightOut = monoRight.requestFullResolutionOutput()
    monoLeftOut.link(stereo.left)
    monoRightOut.link(stereo.right)

    stereo.setRectification(True)
    stereo.setExtendedDisparity(True)
    stereo.setLeftRightCheck(True)

    monoQueue = monoLeftOut.createOutputQueue(maxSize=4)
    monoQueueR = monoRightOut.createOutputQueue(maxSize=4)
    rgbQueue = rgbOut.createOutputQueue(maxSize=4)
    dispQueue = stereo.disparity.createOutputQueue()

# --------------Example usage--------------
    with pipeline:
        pipeline.start()

        while pipeline.isRunning():
            rgb = rgbQueue.get().getFrame()
            cv2.imshow("rgb", rgb)
            key = cv2.waitKey(1)
            if key == PAUSE_KEY:
                cv2.waitKey(0)
            if key == ESC_KEY:
                pipeline.stop()
                break
# --------------------------------------------

else:
    print(f"Loading {NUM_IMAGES} images from {IMAGE_DIR}...")

    image_paths = sorted(IMAGE_DIR.glob(f"*{IMAGE_EXT}"))

    for img_path in image_paths[:NUM_IMAGES]:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Warning: Could not load {img_path}")
            continue

        frames.append(frame)

# ---------------Example usage--------------
    for i, frame in enumerate(frames):
        print(f"Frame {i} shape: {frame.shape}")

# ------------------------------------------

print('Done!')