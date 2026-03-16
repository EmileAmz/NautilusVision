import cv2
import depthai as dai
import numpy as np

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

colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
colorMap[0] = [0, 0, 0]  # to make zero-disparity pixels black

monoLeftQueue = monoLeftOut.createOutputQueue(maxSize=4)

with pipeline:
    pipeline.start()
    maxDisparity = 1

    while pipeline.isRunning():
        disparity = disparityQueue.get()
        assert isinstance(disparity, dai.ImgFrame)
        npDisparity = disparity.getFrame()
        maxDisparity = max(maxDisparity, np.max(npDisparity))
        colorizedDisparity = cv2.applyColorMap(((npDisparity / maxDisparity) * 255).astype(np.uint8), colorMap)

        monoFrame = monoLeftQueue.get().getFrame()  # grayscale
        edges = cv2.Canny(monoFrame, 100, 200)

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

        # ---- Mask depth color by edges ----
        coloredEdges = np.zeros_like(colorizedDisparity)
        coloredEdges[edges > 0] = colorizedDisparity[edges > 0]

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(coloredEdges, (x, y), (x + w, y + h), (0, 255, 0), 1)

        cv2.imshow("disparity", coloredEdges)
        if cv2.waitKey(1) == 27:  # ESC to exit
            pipeline.stop()
            break