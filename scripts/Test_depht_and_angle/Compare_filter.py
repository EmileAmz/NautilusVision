from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

def apply_median_filter(depth, kernel_size):
    return median_filter(depth, size=kernel_size)


def load_depth(depth_path):
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise ValueError("Image non chargée")

    depth = np.squeeze(depth)

    if len(depth.shape) != 2:
        raise ValueError(f"Image non 2D: {depth.shape}")

    print("min raw:", np.min(depth))
    print("max raw:", np.max(depth))

    return depth


def apply_gaussian_filter(depth, kernel_size):
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("Le kernel du filtre gaussien doit être impair et >= 1")
    return cv2.GaussianBlur(depth, (kernel_size, kernel_size), 0)


def compare_filters(depth_path, kernel_sizes, max_depth=5000):
    depth = load_depth(depth_path)
    depth = depth.astype(np.float32)

    results = [("Original", depth)]

    for k in kernel_sizes:
        median_filtered = apply_median_filter(depth, k)
        gaussian_filtered = apply_gaussian_filter(depth, k).astype(np.float32)

        results.append((f"Median k={k}", median_filtered))
        results.append((f"Gaussian k={k}", gaussian_filtered))

    n = len(results)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    for i, (title, img) in enumerate(results):
        img_clipped = np.clip(img, 0, max_depth)

        axes[i].imshow(img_clipped, cmap="plasma", vmin=0, vmax=max_depth)
        axes[i].set_title(title, fontsize=12, pad=10)  # 👈 espace au-dessus
        axes[i].axis("off")

    # cacher les cases vides
    for j in range(len(results), len(axes)):
        axes[j].axis("off")

    # 👇 IMPORTANT : espace entre les lignes
    plt.subplots_adjust(hspace=0.4, wspace=0.1)

    plt.show()

if __name__ == "__main__":
    depth_path = Path(r"C:\Users\Xavier Lefebvre\Documents\dataset\depth\1775077000.068.png")

    kernel_sizes = [3, 5, 7, 9]

    compare_filters(depth_path, kernel_sizes, max_depth=5000)