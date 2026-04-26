import numpy as np
from scipy.ndimage import median_filter
from scipy import ndimage

# PARAMETRES CAMERA
cx = 623
fx = 728
fy = 726
cy = 370


# ---------------- FILL ZEROS ----------------
def fill_zeros_with_nearest_fast(depth_patch):
    depth = depth_patch.copy()

    zero_mask = (depth == 0)

    if not np.any(zero_mask):
        return depth

    if np.all(zero_mask):
        return depth

    _, indices = ndimage.distance_transform_edt(
        zero_mask,
        return_indices=True
    )

    filled = depth.copy()
    filled[zero_mask] = depth[indices[0][zero_mask], indices[1][zero_mask]]

    return filled


# ---------------- FILTER PATCH ----------------
def filter_depth_bbox(depth_frame, x1, y1, x2, y2, kernel_size, margin=5):
    frame_h, frame_w = depth_frame.shape

    # zone élargie
    x1_big = max(0, int(x1 - margin))
    y1_big = max(0, int(y1 - margin))
    x2_big = min(frame_w, int(x2 + margin))
    y2_big = min(frame_h, int(y2 + margin))

    bbox_patch = depth_frame[y1_big:y2_big, x1_big:x2_big]

    if bbox_patch.size == 0:
        return None

    # fill trous
    bbox_patch_filled = fill_zeros_with_nearest_fast(bbox_patch)

    # filtre médian
    bbox_filtered = median_filter(
        bbox_patch_filled,
        size=kernel_size,
        mode='nearest'
    )

    # revenir à la zone centrale
    local_x1 = x1 - x1_big
    local_y1 = y1 - y1_big
    local_x2 = local_x1 + (x2 - x1)
    local_y2 = local_y1 + (y2 - y1)

    center_patch_filtered = bbox_filtered[
        local_y1:local_y2,
        local_x1:local_x2
    ]

    return center_patch_filtered

# ---------------- DEPTH ----------------
def find_depth(depth_frame, half, cy_boundingbox, cx_boundingbox, kernel_size):
    frame_h, frame_w = depth_frame.shape

    y1 = max(0, int(cy_boundingbox - half))
    y2 = min(frame_h, int(cy_boundingbox + half + 1))
    x1 = max(0, int(cx_boundingbox - half))
    x2 = min(frame_w, int(cx_boundingbox + half + 1))

    center_patch = filter_depth_bbox(depth_frame,x1,y1,x2,y2,kernel_size)

    if center_patch is None or center_patch.size == 0:
        return -1000

    center_depth = float(np.median(center_patch))

    if np.isnan(center_depth) or center_depth == 0:
        return -1000

    return int(center_depth)

# ---------------- ANGLE ----------------
def find_angle(box, depth_mean):
    if depth_mean is None or depth_mean <= 0:
        return None

    x_center = box["cx"]

    X = (x_center - cx) * depth_mean / fx

    yaw = np.arctan2(X, depth_mean)
    yaw_deg = np.degrees(yaw)

    return yaw_deg