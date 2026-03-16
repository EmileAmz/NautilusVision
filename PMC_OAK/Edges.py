import cv2
import depthai as dai
import numpy as np
import datetime

# Create pipeline
pipeline = dai.Pipeline()

# Create the Camera node
cam = pipeline.create(dai.node.Camera).build()

# Request an RGB output
rgbOutput = cam.requestOutput(
    size=(640, 360),  # output size
    type=dai.ImgFrame.Type.BGR888i
)

# Create an output queue for the RGB stream
rgbQueue = rgbOutput.createOutputQueue(maxSize=4)

# Start the pipeline
pipeline.start()

while pipeline.isRunning():
    # Get next frame (blocking)
    inRgb = rgbQueue.get()  # ✅ No "blocking=True" keyword

    frame = inRgb.getCvFrame()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Show original + edges
    # Convert edges to BGR so it has 3 channels
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((frame, edges_bgr))

    cv2.imshow("Edge Detection", combined)

    if cv2.waitKey(1) == 27:  # ESC to exit
        pipeline.stop()
        break

cv2.destroyAllWindows()
