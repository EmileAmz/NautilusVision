#!/usr/bin/env python3

import cv2
import depthai as dai
import time

FPS = 30
RGB_SIZE = (1280, 704)
ROTATE_180 = True


def get_latest(queue):
    latest = None
    while True:
        pkt = queue.tryGet()
        if pkt is None:
            break
        latest = pkt
    return latest


def create_oak1_pipeline(pipeline):
    cam = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_A
    )

    rgb_out = cam.requestOutput(
        size=RGB_SIZE,
        fps=FPS,
        type=dai.ImgFrame.Type.NV12
    )

    rgb_queue = rgb_out.createOutputQueue(
        maxSize=1,
        blocking=False
    )

    return rgb_queue


def main():
    pipeline = dai.Pipeline()

    rgb_queue = create_oak1_pipeline(pipeline)

    pipeline.start()

    print("OAK-1 pipeline started")
    print("Press 'q' to quit, 's' to save image")

    last_time = time.time()
    fps_display = 0.0

    while pipeline.isRunning():
        rgb_pkt = get_latest(rgb_queue)

        if rgb_pkt is None:
            continue

        frame = rgb_pkt.getCvFrame()

        if ROTATE_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        now = time.time()
        dt = now - last_time
        last_time = now

        if dt > 0:
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / dt)

        cv2.putText(
            frame,
            f"FPS: {fps_display:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("OAK-1", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            filename = "oak1_snapshot.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()