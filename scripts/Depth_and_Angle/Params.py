import depthai as dai
import numpy as np

with dai.Device() as device:
    calib = device.readCalibration()

    K_rgb = np.array(
        calib.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A,   # ou RGB selon ta version
            1280, 720
        )
    )

    fx = K_rgb[0, 0]
    fy = K_rgb[1, 1]
    cx = K_rgb[0, 2]
    cy = K_rgb[1, 2]

    print(K_rgb)
    print("cx = ", cx)
    print("fx = ", fx)
    print("fy = ", fy)
