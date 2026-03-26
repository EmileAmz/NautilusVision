import cv2
import depthai as dai
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ---------------- CONFIG ----------------
USE_CAMERA = False  # True = camera, False = folder
SCRIPT_DIR = Path(__file__).parent.resolve()
IMAGE_DIR = SCRIPT_DIR / "datasets/Test_Piscine_a_annoter/Tests_march_18/rgb"
LABEL_DIR = SCRIPT_DIR / "datasets/Test_Piscine_a_annoter/Tests_march_18/labels"
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
        img_float = frame.copy().astype(np.float32)
        # Renormalization
        # b, g, r = cv2.split(frame)
        #
        # b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
        # g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
        # r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
        #
        # enhanced = cv2.merge([b, g, r])
        #
        # enhanced = enhanced.astype(np.uint8)


        # Weighted RG canals
        # blue = img_float[:, :, 0]
        # green = img_float[:, :, 1]
        # red = img_float[:, :, 2]
        #
        # # Adjust each channel
        # blue *= 0.6  # reduce blue dominance
        # green *= 1.1  # slight boost
        # red *= 1.8  # strong boost (compensate absorption)
        #
        # # Recombine
        # enhanced = np.stack([blue, green, red], axis=2)
        #
        # # Clip to valid range
        # enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)


        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---- PLOT ----
        # plt.figure()  # smaller display
        #
        # plt.subplot(1, 2, 1)
        # plt.imshow(frame_rgb)
        # plt.title("Original")
        # plt.axis('off')
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(enhanced)
        # plt.title("Enhanced (Weighted RG)")
        # plt.axis('off')
        #
        # plt.tight_layout()
        # plt.show()



        edges = cv2.Canny(frame, 100, 200)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(edges, (x, y), (x + w, y + h), (0, 255, 0), 1)

        plt.figure()  # smaller display

        plt.subplot(1, 2, 1)
        plt.imshow(frame_rgb)
        plt.title("Original")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(edges)
        plt.title("Edges")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        print(f"Frame {i} shape: {frame.shape}")

# ------------------------------------------
cv2.destroyAllWindows()