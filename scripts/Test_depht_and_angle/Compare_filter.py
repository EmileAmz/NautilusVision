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


def create_rbf_kernel(kernel_size, sigma):
    """
    Noyau RBF spatial:
        exp(-r^2 / (2*sigma^2))
    """
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("Le kernel_size doit être impair et >= 1")
    if sigma <= 0:
        raise ValueError("sigma doit être > 0")

    radius = kernel_size // 2
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    r2 = x ** 2 + y ** 2

    kernel = np.exp(-r2 / (2 * sigma ** 2))
    kernel /= np.sum(kernel)

    return kernel.astype(np.float32)


def apply_rbf_filter(depth, kernel_size, sigma):
    """
    Filtre RBF purement spatial.
    """
    kernel = create_rbf_kernel(kernel_size, sigma)
    return cv2.filter2D(depth, -1, kernel)


def apply_rbf_bilateral_filter(depth, kernel_size, sigma_spatial, sigma_range):
    """
    Filtre RBF edge-preserving (style bilateral).

    - sigma_spatial contrôle l'influence de la distance dans l'image
    - sigma_range contrôle l'influence de la différence de profondeur

    Plus sigma_range est petit, plus les bords sont préservés.
    """
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("Le kernel_size doit être impair et >= 1")
    if sigma_spatial <= 0 or sigma_range <= 0:
        raise ValueError("sigma_spatial et sigma_range doivent être > 0")

    depth = depth.astype(np.float32)
    h, w = depth.shape
    radius = kernel_size // 2

    padded = np.pad(depth, radius, mode="reflect")
    filtered = np.zeros_like(depth, dtype=np.float32)

    # poids spatial fixe
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    spatial_weights = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_spatial ** 2)).astype(np.float32)

    for i in range(h):
        for j in range(w):
            local_patch = padded[i:i + kernel_size, j:j + kernel_size]
            center_value = padded[i + radius, j + radius]

            # poids sur la différence de profondeur
            range_weights = np.exp(-((local_patch - center_value) ** 2) / (2 * sigma_range ** 2)).astype(np.float32)

            # combinaison spatiale + profondeur
            weights = spatial_weights * range_weights
            weights_sum = np.sum(weights)

            if weights_sum > 1e-8:
                filtered[i, j] = np.sum(weights * local_patch) / weights_sum
            else:
                filtered[i, j] = center_value

    return filtered


def compare_filters(depth_path, kernel_sizes, max_depth=5000,
                    rbf_sigma_factor=0.3,
                    bilateral_spatial_factor=0.3,
                    bilateral_range_sigma=150):
    depth = load_depth(depth_path)
    depth = depth.astype(np.float32)

    results = [("Original", depth)]

    for k in kernel_sizes:
        median_filtered = apply_median_filter(depth, k).astype(np.float32)
        gaussian_filtered = apply_gaussian_filter(depth, k).astype(np.float32)

        sigma_rbf = max(1.0, k * rbf_sigma_factor)
        rbf_filtered = apply_rbf_filter(depth, k, sigma_rbf).astype(np.float32)

        sigma_spatial = max(1.0, k * bilateral_spatial_factor)
        rbf_bilateral_filtered = apply_rbf_bilateral_filter(
            depth,
            kernel_size=k,
            sigma_spatial=sigma_spatial,
            sigma_range=bilateral_range_sigma
        ).astype(np.float32)

        results.append((f"Median k={k}", median_filtered))
        results.append((f"Gaussian k={k}", gaussian_filtered))
        results.append((f"RBF k={k}, σ={sigma_rbf:.1f}", rbf_filtered))
        results.append((
            f"RBF edge-preserving k={k}\nσs={sigma_spatial:.1f}, σr={bilateral_range_sigma}",
            rbf_bilateral_filtered
        ))

    n = len(results)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    for i, (title, img) in enumerate(results):
        img_clipped = np.clip(img, 0, max_depth)
        axes[i].imshow(img_clipped, cmap="plasma", vmin=0, vmax=max_depth)
        axes[i].set_title(title, fontsize=11, pad=10)
        axes[i].axis("off")

    for j in range(len(results), len(axes)):
        axes[j].axis("off")

    plt.subplots_adjust(hspace=0.6, wspace=0.15)
    plt.show()


if __name__ == "__main__":
    depth_path = Path(r"C:\Users\Xavier Lefebvre\Documents\dataset\depth\1775077000.068.png")

    kernel_sizes = [3, 5, 7]

    compare_filters(
        depth_path,
        kernel_sizes,
        max_depth=5000,
        rbf_sigma_factor=0.3,
        bilateral_spatial_factor=0.3,
        bilateral_range_sigma=150
    )