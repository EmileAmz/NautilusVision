#!/usr/bin/env python3

import cv2
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Index caméra, ex: 0, 1, 2")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save", action="store_true", help="Enregistre la vidéo")
    args = parser.parse_args()

    # Linux/Jetson: CAP_V4L2 aide souvent avec les caméras USB
    #cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"Erreur: impossible d'ouvrir /dev/video{args.camera}")
        return

    # Important pour USB2.0: MJPEG réduit beaucoup la bande passante
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Caméra ouverte: {real_w}x{real_h} @ {real_fps:.1f} FPS")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter("elp_camera_output.avi", fourcc, args.fps, (real_w, real_h))

    prev_time = time.time()
    fps_display = 0.0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erreur: frame non reçue")
            break

        now = time.time()
        dt = now - prev_time
        prev_time = now

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

        cv2.imshow("ELP USB Fisheye Camera", frame)

        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            cv2.imwrite("elp_snapshot.jpg", frame)
            print("Image sauvegardée: elp_snapshot.jpg")

    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()