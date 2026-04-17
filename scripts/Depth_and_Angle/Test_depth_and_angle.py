import numpy as np
import cv2
from scripts.Normalisation_depth import filter_depth

#PARAMETRES DE LA CAMERA
cx = 623
fx = 728
fy = 726
cy = 370

def find_depth(depth_path, kernel_size, half, box, image_width, image_height, use_filtered_depth=False):
    # Charger la depth map
    if use_filtered_depth:
        depth = filter_depth(depth_path, kernel_size)
    else:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise ValueError(f"Impossible de charger l'image depth : {depth_path}")

    depth = np.squeeze(depth)

    if len(depth.shape) != 2:
        raise ValueError(f"Depth non 2D après squeeze : {depth.shape}")

    x_center = int(box["cx"])
    y_center = int(box["cy"])

    # Patch autour du centre
    x1 = max(0, x_center - half)
    x2 = min(image_width, x_center + half)
    y1 = max(0, y_center - half)
    y2 = min(image_height, y_center + half)

    patch = depth[y1:y2, x1:x2]

    if patch.size == 0:
        return None

    valid_patch = patch[patch > 0]

    if valid_patch.size == 0:
        return None

    Z = float(np.median(valid_patch))

    # Calcul vraie distance
    X = (x_center - cx) * Z / fx
    Y = (y_center - cy) * Z / fy

    true_distance = np.sqrt(X**2 + Y**2 + Z**2)

    return true_distance


def find_angle(box, depth_mean):
    if depth_mean is None:
        return None

    x_center = box["cx"]

    # Coordonnée horizontale dans le repère caméra
    X = (x_center - cx) * depth_mean / fx

    # Yaw pour aligner l'objet avec le centre
    yaw = np.arctan2(X, depth_mean)
    yaw_deg = np.degrees(yaw)

    return yaw_deg