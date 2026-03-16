import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_object_pose_grabcut(frame, depth_frame, bbox, fx, fy, cx, cy):
    x1, y1, x2, y2 = map(int, bbox)
    roi = frame[y1:y2, x1:x2]
    depth_roi = depth_frame[y1:y2, x1:x2]

    h, w = roi.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    rect = (5, 5, w-10, h-10)

    cv2.grabCut(roi, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    # convert mask to binary foreground
    mask_fg = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')

    # apply mask to depth
    object_depth = depth_roi[mask_fg == 1]
    object_depth = object_depth[object_depth > 0]

    if len(object_depth) == 0:
        return None, None, None, mask_fg

    distance = np.median(object_depth)
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    horizontal_angle = np.arctan((x_center - cx) / fx)
    vertical_angle = np.arctan((y_center - cy) / fy)

    return distance, horizontal_angle, vertical_angle, mask_fg

def get_object_3d_vectorized(depth_frame, bbox, fx, fy, cx, cy):
    x1, y1, x2, y2 = map(int, bbox)
    depth_crop = depth_frame[y1:y2, x1:x2].astype(np.float32) / 1000.0

    h, w = depth_crop.shape
    u = np.arange(x1, x2)
    v = np.arange(y1, y2)
    uu, vv = np.meshgrid(u, v)

    # flatten everything
    Z = depth_crop.flatten()
    uu = uu.flatten()
    vv = vv.flatten()

    # keep valid depth pixels
    valid = (Z > 0.5) & (Z < 4.0)
    Z = Z[valid]
    uu = uu[valid]
    vv = vv[valid]

    if Z.size == 0:
        return None

    # convert pixels → 3D coordinates
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    points = np.stack((X, Y, Z), axis=1)

    # remove background cluster
    z_threshold = np.percentile(points[:, 2], 40)
    object_points = points[points[:, 2] <= z_threshold]

    if object_points.shape[0] == 0:
        return None

    median_point = np.median(object_points, axis=0)
    return median_point, points

def plot_depth_histogram(grabcut_depth, vector_depth):
    plt.figure(figsize=(8,4))
    plt.hist(grabcut_depth, bins=50, alpha=0.6, label='GrabCut depth')
    plt.hist(vector_depth, bins=50, alpha=0.6, label='Vectorized depth')
    plt.title('Depth distribution inside bounding box')
    plt.xlabel('Depth (meters)')
    plt.ylabel('Pixel count')
    plt.legend()
    plt.show()

def show_object_overlay(frame, bbox, grabcut_mask, vector_points, cx, cy, fx, fy):
    overlay_gc = frame.copy()
    h, w = grabcut_mask.shape
    x1, y1, x2, y2 = bbox
    # GrabCut overlay (green)
    overlay_gc[y1:y2, x1:x2, 1] = np.where(grabcut_mask==1, 255, overlay_gc[y1:y2, x1:x2, 1])

    # Vectorized overlay (red)
    overlay_vec = frame.copy()
    for pt in vector_points:
        x_pix = int(pt[0]*fx/pt[2] + cx)
        y_pix = int(pt[1]*fy/pt[2] + cy)
        if 0 <= x_pix < frame.shape[1] and 0 <= y_pix < frame.shape[0]:
            overlay_vec[y_pix, x_pix, 0] = 255  # Red channel

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.imshow(overlay_gc)
    plt.title("GrabCut object overlay")
    plt.subplot(1,2,2)
    plt.imshow(overlay_vec)
    plt.title("Vectorized object overlay")
    plt.show()

if __name__ == '__main__':
    xmin, ymin, xmax, ymax = 300, 100, 900, 600
    bbox = (xmin, ymin, xmax, ymax)
    fx, fy, cx, cy = 870, 870, 640, 360

    frame = cv2.imread('C:/Users/eaime/Documents/dataset/rgb/1773337279.884.png')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    depth_frame = cv2.imread("C:/Users/eaime/Documents/dataset/depth/1773337279.884.png",cv2.IMREAD_UNCHANGED)

    # --- GrabCut ---
    distance, h_angle, v_angle, mask_fg = get_object_pose_grabcut(
        frame, depth_frame, bbox, fx, fy, cx, cy
    )
    print("GrabCut distance:", distance)
    print("Horizontal angle (deg):", np.degrees(h_angle))
    print("Vertical angle (deg):", np.degrees(v_angle))

    # Depths for histogram
    depth_crop = depth_frame[ymin:ymax, xmin:xmax].astype(np.float32)/1000.0
    grabcut_depth = depth_crop[mask_fg==1]
    grabcut_depth = grabcut_depth[grabcut_depth>0]

    # --- Vectorized ---
    result = get_object_3d_vectorized(depth_frame, bbox, fx, fy, cx, cy)
    if result is not None:
        median_point, points = result
        vector_depth = points[:,2]
        X, Y, Z = median_point
        print("Vectorized median object position (camera frame):")
        print("X:", X, "Y:", Y, "Z:", Z)
    else:
        vector_depth = np.array([])

    # --- Histograms ---
    plot_depth_histogram(grabcut_depth, vector_depth)

    # --- Overlay images ---
    if result is not None:
        show_object_overlay(frame, bbox, mask_fg, points, cx, cy, fx, fy)