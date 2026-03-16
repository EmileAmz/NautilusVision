import cv2
import depthai as dai
import numpy as np

# ---------------- Pipeline ----------------
pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
stereo = pipeline.create(dai.node.StereoDepth)

# Linking
monoLeftOut = monoLeft.requestFullResolutionOutput()
monoRightOut = monoRight.requestFullResolutionOutput()
monoLeftOut.link(stereo.left)
monoRightOut.link(stereo.right)

stereo.setRectification(True)
stereo.setExtendedDisparity(True)
stereo.setLeftRightCheck(True)

disparityQueue = stereo.disparity.createOutputQueue()
monoLeftQueue = monoLeftOut.createOutputQueue(maxSize=4)

# ---------------- Run ----------------
with pipeline:
    pipeline.start()
    maxDisparity = 1

    while pipeline.isRunning():

        # ---- Get frames ----
        disparity = disparityQueue.get()
        monoFrame = monoLeftQueue.get().getFrame()

        npDisparity = disparity.getFrame()
        maxDisparity = max(maxDisparity, np.max(npDisparity))

        # ---- Depth mask (optional for whiteboard testing) ----
        # Set useDepthMask = False to ignore depth
        useDepthMask = False

        if useDepthMask:
            depthMask = npDisparity > 0
            minDisp = 20    # tune for underwater
            maxDisp = 200
            depthMask &= (npDisparity > minDisp) & (npDisparity < maxDisp)
            depthMask = depthMask.astype(np.uint8) * 255
            kernel = np.ones((5,5), np.uint8)
            depthMask = cv2.morphologyEx(depthMask, cv2.MORPH_CLOSE, kernel)
            depthMask = cv2.morphologyEx(depthMask, cv2.MORPH_OPEN, kernel)

        # ---- Edge detection ----
        edges = cv2.Canny(monoFrame, 30, 100)  # lower thresholds for drawings
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        edgeMask = (edges > 0).astype(np.uint8) * 255

        # ---- Combine depth + edges ----
        if useDepthMask:
            combinedMask = cv2.bitwise_and(depthMask, edgeMask)
        else:
            combinedMask = edgeMask.copy()  # just use edges for whiteboard
        combinedMask = cv2.morphologyEx(
            combinedMask,
            cv2.MORPH_CLOSE,
            np.ones((7,7), np.uint8)
        )

        # ---- Find contours ----
        contours, _ = cv2.findContours(
            combinedMask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # ---- Prepare debug view ----
        debug = cv2.cvtColor(monoFrame, cv2.COLOR_GRAY2BGR)

        # ---- Process each contour / object ----
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 200:  # smaller threshold for drawings
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            if aspect < 0.1 or aspect > 6.0:
                continue

            # Optional depth-normalized score for underwater
            meanDisp = 1
            if useDepthMask:
                mask_temp = np.zeros(npDisparity.shape, np.uint8)
                cv2.drawContours(mask_temp, [cnt], -1, 255, -1)
                meanDisp = np.mean(npDisparity[mask_temp > 0])
                score = area * (meanDisp ** 2)
                if score < 1e6 or score > 1e9:
                    continue

            # ---- Isolate object ----
            objMask = np.zeros(monoFrame.shape, np.uint8)
            cv2.drawContours(objMask, [cnt], -1, 255, -1)
            isolated = cv2.bitwise_and(monoFrame, monoFrame, mask=objMask)
            crop = isolated[y:y+h, x:x+w]

            # ---- Resize / pad for display ----
            max_size = 516
            scale = max_size / max(h, w)
            new_h, new_w = int(h*scale), int(w*scale)
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            padded = np.zeros((max_size, max_size), dtype=crop.dtype)
            padded[:new_h, :new_w] = resized

            cv2.imshow(f"isolated_{idx}", padded)
            # Draw rectangle on debug view
            cv2.rectangle(debug, (x,y), (x+w, y+h), (0,255,0), 2)

        # ---- Show debug views ----
        cv2.imshow("mono", monoFrame)
        if useDepthMask:
            cv2.imshow("depthMask", depthMask)
        cv2.imshow("edges", edges)
        cv2.imshow("combined", combinedMask)
        cv2.imshow("debug", debug)

        if cv2.waitKey(1) == 27:  # ESC to exit
            pipeline.stop()
            break
