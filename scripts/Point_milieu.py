import numpy as np
from pathlib import Path
import cv2
from scripts.Normalisation_depth import filter_depth





def find_depth(results_model, model, kernel_size, half):
    for r in results_model:
        angle = 0
        img = r.orig_img  # image numpy, généralement en BGR
        h, w = img.shape[:2]
        print(f"\nImage : {r.path}")

        r_path = Path(r.path)

        # Récupérer seulement le nom du fichier
        filename = r_path.name

        # Nouveau dossier où sont les images
        IMAGE_DIR = Path("datasets/Test_Piscine_a_annoter/Tests_march_18/depth")

        # Reconstruire le bon chemin
        img_path = IMAGE_DIR / filename

        # Lire l'image
        #img_profondeur = filter_depth(img_path, kernel_size)
        img_profondeur = cv2.imread(str(img_path))

        for i, box in enumerate(r.boxes):
            x_center, y_center, bw, bh = box.xywh[0]

            # convertir en int pour indexer dans l'image
            x_center = int(round(x_center.item()))
            y_center = int(round(y_center.item()))

            # bornes sécurisées pour rester dans l'image
            x1 = max(0, x_center - half)
            x2 = min(w, x_center + half)
            y1 = max(0, y_center - half)
            y2 = min(h, y_center + half)

            # patch 10x10 autour du centre
            patch = img_profondeur[y1:y2, x1:x2]

            if patch.size == 0:
                print(f"Box {i}: patch vide")
                continue

            # canal rouge en BGR = index 2
            depth_mean = np.mean(patch[:, :, 2])

            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            angle = find_angle(x1, y1, depth_mean)

            print(
                f"Box {i} | classe={model.names[cls]} | conf={conf:.2f} | "
                f"centre=({x_center},{y_center}) | profondeur={depth_mean:.2f} | angle={angle:.2f}"
            )
    return

def find_angle(x_center, y_center, depth_mean):

    yaw = 0
    X = 0

    #À trouver
    fx = 494
    fy = 499
    cx = 321
    cy = 218

    X = (x_center - cx) * depth_mean / fx
    yaw = np.arctan2(X, depth_mean)
    yaw_deg = np.degrees(yaw)

    return yaw_deg