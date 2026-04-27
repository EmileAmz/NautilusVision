import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import median_filter

def vertical_median_filter(img, ksize=5):
    pad = ksize // 2
    padded = np.pad(img, ((pad, pad), (0, 0)), mode='edge')
    out = np.empty_like(img)

    for y in range(img.shape[0]):
        window = padded[y:y+ksize, :]
        out[y, :] = np.median(window, axis=0)

    return out
def median_filter_2d(img, ksize):
    if ksize % 2 == 0:
        ksize += 1  # force impair automatiquement

    img = np.squeeze(img)

    if img.ndim != 2:
        raise ValueError(f"Image non 2D après squeeze: {img.shape}")

    # Vérification debug (optionnel)
    # print("dtype:", img.dtype)

    return median_filter(img, size=ksize)

def median_filter_ignore_zeros(img, ksize, zero_threshold=0.8):
    if ksize % 2 == 0:
        ksize += 1  # force impair

    img = np.squeeze(img)

    if img.ndim != 2:
        raise ValueError(f"Image non 2D après squeeze: {img.shape}")

    h, w = img.shape
    r = ksize // 2

    padded = np.pad(img, r, mode='reflect')
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            patch = padded[i:i+ksize, j:j+ksize]

            # ratio de zéros
            zero_ratio = np.sum(patch == 0) / patch.size

            if zero_ratio >= zero_threshold:
                output[i, j] = 0
            else:
                # enlever les zéros
                valid_values = patch[patch != 0]

                if valid_values.size > 0:
                    output[i, j] = np.median(valid_values)
                else:
                    output[i, j] = 0  # fallback sécurité

    return output.astype(img.dtype)

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
        #depth_filtered = median_filter_ignore_zeros(depth, KERNEL_SIZE)
        #depth_filtered = apply_rbf_bilateral_filter(depth_filtered, KERNEL_SIZE,1,150 )

    else:
        raise ValueError(f"Mode inconnu: {FILTER_MODE}")

    return depth_filtered

def global_median_ignore_zeros(img):
    img = np.squeeze(img)

    if img.ndim != 2:
        raise ValueError(f"Image non 2D: {img.shape}")

    # garder seulement les valeurs non nulles
    valid_values = img[img != 0]

    if valid_values.size == 0:
        return 0  # fallback si tout est zéro

    return np.median(valid_values)

if __name__ == "__main__":
    #depth_path = Path(r"C:\Users\Xavier Lefebvre\Documents\dataset\depth\1775152864.904.png")
    depth_path = Path(r"C:\Users\Xavier Lefebvre\Documents\dataset\depth_prequalif\20260422_105233.png")
    depth_filtered = filter_depth(depth_path, 7)

    # ---------------- CLEAN ----------------
    depth_filtered = depth_filtered.astype(np.float32)

    # Remplacer valeurs invalides (0) par NaN pour affichage
    #depth_filtered[depth_filtered == 0] = np.nan

    # ---------------- CLIP à 4000 mm ----------------
    MAX_DEPTH = 5000  # mm (4 m)
    depth_clipped = np.clip(depth_filtered, 0, MAX_DEPTH)

    # ---------------- PLOT ----------------
    plt.figure(figsize=(10, 6))

    # Créer un masque des zéros
    mask_zero = (depth_clipped == 0)

    # Mettre les zéros à NaN (matplotlib ne les affiche pas)
    depth_display = depth_clipped.astype(np.float32)
    depth_display[mask_zero] = np.nan

    # Créer la colormap
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color='black')  # 🔥 les NaN deviennent noirs

    median_value = global_median_ignore_zeros(depth_filtered)
    print("Median (sans zéros):", median_value)

    # Affichage
    im = plt.imshow(depth_display, cmap=cmap, vmin=0, vmax=MAX_DEPTH)

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

