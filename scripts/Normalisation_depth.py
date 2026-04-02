import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def vertical_median_filter(img, ksize=5):
    pad = ksize // 2
    padded = np.pad(img, ((pad, pad), (0, 0)), mode='edge')
    out = np.empty_like(img)

    for y in range(img.shape[0]):
        window = padded[y:y+ksize, :]
        out[y, :] = np.median(window, axis=0)

    return out

def median_filter_2d(img, ksize):
    return cv2.medianBlur(img, ksize)

def filter_depth(depth_path, kernel_size):

# ---------------- CONFIG ----------------
#depth_path = Path(r"C:\Users\Xavier Lefebvre\Documents\GitHub\NautilusVision\datasets\Test_Piscine_a_annoter\Tests_march_18\depth\1773859763.894.png")

    FILTER_MODE = "2d"
    # options: "none", "vertical", "horizontal", "2d"

    KERNEL_SIZE = kernel_size

    # ---------------- LOAD ----------------
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise ValueError("Image non chargée")

    depth = np.squeeze(depth)

    if len(depth.shape) != 2:
        raise ValueError(f"Image non 2D: {depth.shape}")

    print("min raw:", np.min(depth))
    print("max raw:", np.max(depth))

    # ---------------- INVERSION ----------------
    # car chez toi: proche = petite valeur → on inverse
    #depth = depth.max() - depth

    # ---------------- FILTER SWITCH ----------------
    if FILTER_MODE == "none":
        depth_filtered = depth

    elif FILTER_MODE == "vertical":
        #depth_filtered = cv2.GaussianBlur(depth, (1, KERNEL_SIZE), 0)
        depth_filtered = vertical_median_filter(depth, KERNEL_SIZE)

    elif FILTER_MODE == "horizontal":
        depth_filtered = cv2.GaussianBlur(depth, (KERNEL_SIZE, 1), 0)

    elif FILTER_MODE == "2d":
        #depth_filtered = cv2.GaussianBlur(depth, (KERNEL_SIZE, KERNEL_SIZE), 0)
        depth_filtered = median_filter_2d(depth, KERNEL_SIZE)

    else:
        raise ValueError(f"Mode inconnu: {FILTER_MODE}")

    return depth_filtered

if __name__ == "__main__":
    depth_path = Path(r"C:\Users\Xavier Lefebvre\Documents\dataset\depth\1775077000.068.png")
    depth_filtered = filter_depth(depth_path, 5)

    # ---------------- CLEAN ----------------
    depth_filtered = depth_filtered.astype(np.float32)

    # Remplacer valeurs invalides (0) par NaN pour affichage
    #depth_filtered[depth_filtered == 0] = np.nan

    # ---------------- CLIP à 4000 mm ----------------
    MAX_DEPTH = 5000  # mm (4 m)
    depth_clipped = np.clip(depth_filtered, 0, MAX_DEPTH)

    # ---------------- PLOT ----------------
    plt.figure(figsize=(10, 6))

    im = plt.imshow(depth_clipped, cmap="plasma", vmin=0, vmax=MAX_DEPTH)

    cbar = plt.colorbar(im)
    cbar.set_label("Profondeur (mm)")

    plt.title("Depth (clipped à 4m)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    """
    plt.figure(figsize=(10, 6))

    im = plt.imshow(depth_filtered, cmap="plasma", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)
    cbar.set_label("Profondeur (raw corrigée)")

    plt.title(f"Depth")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    """

