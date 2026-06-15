import cv2

for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

    if cap.isOpened():
        ret, frame = cap.read()

        if ret:
            print(f"Camera trouvée à l'index {i}")
        else:
            print(f"Index {i} ouvert mais aucune image")

    cap.release()