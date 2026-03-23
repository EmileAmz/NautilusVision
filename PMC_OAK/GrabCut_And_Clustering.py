import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def get_object_pose_grabcut(frame, depth_frame, bbox, fx, fy, cx, cy):
    x1, y1, x2, y2 = map(int, bbox)
    roi = frame[y1:y2, x1:x2]
    depth_roi = depth_frame[y1:y2, x1:x2]

    h, w = roi.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    margin = 20
    rect = (margin, margin, w-2*margin, h-2*margin)

    cv2.grabCut(roi, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask_fg = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')

    kernel = np.ones((5,5), np.uint8)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel)

    object_depth = depth_roi[mask_fg == 1]
    object_depth = object_depth[(object_depth > 0.3) & (object_depth < 2.0)]

    if len(object_depth) == 0:
        return None, None, None, mask_fg

    distance = np.median(object_depth)

    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    horizontal_angle = np.arctan((x_center - cx) / fx)
    vertical_angle = np.arctan((y_center - cy) / fy)

    return distance, horizontal_angle, vertical_angle, mask_fg


def get_object_3d_vectorized(depth_frame, bbox, fx, fy, cx, cy, grabcut_mask=None):
    x1, y1, x2, y2 = map(int, bbox)

    depth_crop = depth_frame[y1:y2, x1:x2]

    # Median filter
    depth_crop = cv2.medianBlur(depth_crop, 5)

    h, w = depth_crop.shape

    u = np.arange(x1, x2)
    v = np.arange(y1, y2)
    uu, vv = np.meshgrid(u, v)

    Z = depth_crop.flatten()
    uu = uu.flatten()
    vv = vv.flatten()

    # 🔥 Relaxed depth filter
    valid = (Z > 0.3) & (Z < 2.0)

    # 🔥 Soft fusion with GrabCut (NOT hard masking)
    if grabcut_mask is not None:
        mask_flat = grabcut_mask.flatten().astype(bool)

        # keep:
        # - GrabCut pixels OR
        # - nearby depth pixels (likely object)
        soft_region = mask_flat | (Z < 0.8)

        valid = valid & soft_region

    Z = Z[valid]
    uu = uu[valid]
    vv = vv[valid]

    print("Points after filtering:", len(Z))

    if Z.size == 0:
        return None

    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    points = np.stack((X, Y, Z), axis=1)

    # 🔥 Relaxed DBSCAN
    clustering = DBSCAN(eps=0.08, min_samples=20).fit(points)
    labels = clustering.labels_

    valid_labels = labels[labels != -1]
    if len(valid_labels) == 0:
        return None

    unique, counts = np.unique(valid_labels, return_counts=True)
    main_cluster = unique[np.argmax(counts)]

    object_points = points[labels == main_cluster]

    print("Points after clustering:", len(object_points))

    if object_points.shape[0] == 0:
        return None

    median_point = np.median(object_points, axis=0)

    return median_point, points, object_points


def plot_depth_histogram(grabcut_depth, vector_depth):
    plt.figure(figsize=(8,4))
    plt.hist(grabcut_depth, bins=50, alpha=0.6, label='GrabCut depth')
    plt.hist(vector_depth, bins=50, alpha=0.6, label='Vectorized depth')
    plt.title('Depth distribution inside bounding box')
    plt.xlabel('Depth (meters)')
    plt.ylabel('Pixel count')
    plt.legend()
    plt.show()


def show_object_overlay(frame, bbox, grabcut_mask, object_points, cx, cy, fx, fy):
    x1, y1, x2, y2 = bbox

    overlay_gc = frame.copy()
    overlay_gc[y1:y2, x1:x2, 0] = np.where(
        grabcut_mask==1, 255, overlay_gc[y1:y2, x1:x2, 0]
    )

    overlay_vec = frame.copy()
    for pt in object_points:
        x_pix = int(pt[0]*fx/pt[2] + cx)
        y_pix = int(pt[1]*fy/pt[2] + cy)
        if 0 <= x_pix < frame.shape[1] and 0 <= y_pix < frame.shape[0]:
            overlay_vec[y_pix, x_pix, 0] = 255

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(overlay_gc)
    plt.title("GrabCut object overlay")

    plt.subplot(1,2,2)
    plt.imshow(overlay_vec)
    plt.title("Vectorized object overlay")

    plt.show()


def visualize_depth_clipped(depth_frame, min_depth=0.2, max_depth=6.0):
    """
    Clip depth values to [min_depth, max_depth], normalize to 0-255, and apply a colormap.
    """
    depth = depth_frame.astype(np.float32)

    # Clip values
    depth_clipped = np.clip(depth, min_depth, max_depth)

    depth_smoothed = cv2.GaussianBlur(cv2.GaussianBlur(depth_clipped,(7,7),1.5),(7,7),1.5)

    # Normalize to 0-255
    depth_normalized = np.uint8(255 * (depth_smoothed - min_depth) / (max_depth - min_depth))

    # Apply color map
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    return depth_colored

if __name__ == '__main__':

    xmin, ymin, xmax, ymax = 200, 100, 1200, 600
    bbox = (xmin, ymin, xmax, ymax)

    fx, fy, cx, cy = 870, 870, 640, 360

    rgbPath = r"C:\Users\eaime\OneDrive - USherbrooke\S7GRO\NautilusVision\datasets\Tests_march_18\dataset\rgb\1773859763.894.png"
    depthPath = r"C:\Users\eaime\OneDrive - USherbrooke\S7GRO\NautilusVision\datasets\Tests_march_18\dataset\depth\1773859763.894.png"

    frame = cv2.imread(rgbPath)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    depth_frame = cv2.imread(depthPath, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    depth_rgb = visualize_depth_clipped(depth_frame, min_depth=0.2, max_depth=6.0)

    plt.figure(figsize=(10, 6))
    plt.imshow(depth_rgb)
    plt.title("Depth Map (Normalized, RGB)")
    plt.show()


    print("RGB shape:", frame.shape)
    print("Depth shape:", depth_frame.shape)
    print("Depth RGB shape", depth_rgb.shape)

    # GrabCut
    distance, h_angle, v_angle, mask_fg = get_object_pose_grabcut(
        frame, depth_frame, bbox, fx, fy, cx, cy
    )

    print("\nGrabCut distance (m):", distance)

    depth_crop = depth_frame[ymin:ymax, xmin:xmax]
    grabcut_depth = depth_crop[mask_fg==1]

    depth_frame_blur = cv2.GaussianBlur(cv2.GaussianBlur(depth_frame,(7,7),1.5),(7,7),1.5)
    # Vectorized
    result = get_object_3d_vectorized(
        depth_frame_blur, bbox, fx, fy, cx, cy, grabcut_mask=mask_fg
    )

    if result is not None:
        median_point, all_points, object_points = result
        vector_depth = object_points[:,2]

        print("\nVectorized median Z:", median_point[2])
    else:
        vector_depth = np.array([])
        object_points = []

    plot_depth_histogram(grabcut_depth, vector_depth)

    if result is not None:
        show_object_overlay(frame, bbox, mask_fg, object_points, cx, cy, fx, fy)