import cv2
import depthai as dai
import numpy as np

# ===================== CONFIG =====================
TEST_STAGE = 1   # <-- CHANGE THIS ONLY
USE_DEPTH = False  # disable for whiteboard testing
PAUSE_KEY = ord('p')
ESC_KEY = 27
# ==================================================

pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
stereo = pipeline.create(dai.node.StereoDepth)

rgbOut = rgb.requestOutput(
    size=(1280,720),
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

with pipeline:
    pipeline.start()

    while pipeline.isRunning():

        mono = monoQueue.get().getFrame()
        monoR = monoQueueR.get().getFrame()
        rgb = rgbQueue.get().getFrame()
        debug = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

        # ---------------- STAGE 1 ----------------
        if TEST_STAGE == 1:
            cv2.imshow("mono", mono)
            cv2.imshow("monoR", monoR)
            cv2.imshow("rgb", rgb)

        # ---------------- STAGE 2 ----------------
        if TEST_STAGE >= 2:
            edges = cv2.Canny(mono, 350, 600)
            if TEST_STAGE == 2:
                cv2.imshow("edges", edges)

        # ---------------- STAGE 3 ----------------
        if TEST_STAGE >= 3:
            edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
            edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
            if TEST_STAGE == 3:
                cv2.imshow("edges+structure", edges)

        # ---------------- STAGE 4 ----------------
        if TEST_STAGE >= 4:
            contours, _ = cv2.findContours(
                edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if TEST_STAGE == 4:
                for cnt in contours:
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(debug, (x,y), (x+w,y+h), (0,255,0), 1)
                cv2.imshow("contours (raw)", debug)

        # ---------------- STAGE 5 ----------------
        if TEST_STAGE >= 5:
            H, W = mono.shape
            imgArea = H * W

            filtered = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 0.002 * imgArea or area > 0.02 * imgArea:
                    continue
                filtered.append(cnt)

            if TEST_STAGE == 5:
                for cnt in filtered:
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(debug, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.imshow("filtered contours", debug)

        # ---------------- STAGE 6 ----------------
        if TEST_STAGE >= 6:
            for i, cnt in enumerate(filtered):
                x,y,w,h = cv2.boundingRect(cnt)

                mask = np.zeros(mono.shape, np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                isolated = cv2.bitwise_and(mono, mono, mask=mask)
                crop = isolated[y:y+h, x:x+w]

                # normalize to fixed size
                size = 128
                scale = size / max(w, h)
                new_w, new_h = int(w*scale), int(h*scale)
                resized = cv2.resize(crop, (new_w, new_h))

                padded = np.zeros((size, size), dtype=np.uint8)
                padded[:new_h, :new_w] = resized

                cv2.imshow(f"isolated_{i}", padded)

                cv2.rectangle(debug, (x,y), (x+w,y+h), (0,255,0), 2)

            cv2.imshow("final detections", debug)

        # ---------------- Controls ----------------
        key = cv2.waitKey(1)
        if key == PAUSE_KEY:
            cv2.waitKey(0)
        if key == ESC_KEY:
            pipeline.stop()
            break
