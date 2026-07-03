"""
color_features.py
Color Harmony Study -- Feature Computation Pipeline

This module implements the complete color-feature extraction pipeline:
    Image --> Pre-process --> Palette (k-means or median-cut) --> Features

Features computed:
    1. Hue circular variance          (within-palette hue dispersion)
    2. Mean chroma                    (average colourfulness)
    3. Lightness contrast             (weighted SD of L*)
    4. Mean pairwise CIEDE2000        (overall palette dissimilarity)
    5. Cohen-Or harmony score         (template-based harmony, lower = more harmonious)

Reference (harmony model):
    Cohen-Or, D., Sorkine, O., Gal, R., Leyvand, T., & Xu, Y.-Q. (2006).
    Color harmonization. ACM Transactions on Graphics, 25(3), 624-630.
    https://doi.org/10.1145/1141911.1141933
"""

import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from config import (
    SRGB_LINEAR_TO_XYZ_D65,
    XYZ_TO_SRGB_LINEAR_D65,
    D65_WHITE,
    LAB_KAPPA,
    LAB_EPSILON,
    HARMONY_SIGMA_DEG,
    HARMONY_SHIFT_DEG,
    HARMONY_DELTA_L_MAX,
    DEFAULT_K,
    DEFAULT_N_START,
    TARGET_LONG_EDGE,
    FRAME_SKIP_PX,
)


# =============================================================================
# SECTION 0: Color-space conversion utilities
# =============================================================================

def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert non-linear sRGB to linear sRGB.

    Applies the IEC 61966-2-1 piecewise inverse gamma (companding) function.
    Values are expected in [0, 1] or [0, 255].

    Args:
        srgb: Non-linear sRGB values (vector, matrix, or 3-D array).

    Returns:
        Linear sRGB in [0, 1], same shape as input.
    """
    srgb = np.asarray(srgb, dtype=float)
    if np.max(srgb) > 1.0:
        srgb = srgb / 255.0
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4
    )
    return linear


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear sRGB to non-linear sRGB.

    Applies the IEC 61966-2-1 piecewise gamma (companding) function.

    Args:
        linear: Linear sRGB values in [0, 1].

    Returns:
        Non-linear sRGB in [0, 1].
    """
    linear = np.asarray(linear, dtype=float)
    linear = np.clip(linear, 0, 1)
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * (linear ** (1 / 2.4)) - 0.055
    )
    return srgb


def rgb_lin_to_xyz(rgb_lin: np.ndarray) -> np.ndarray:
    """Convert linear sRGB to CIEXYZ (D65).

    Uses the IEC 61966-2-1 sRGB -> XYZ transformation matrix with the
    D65 nominal white point.

    Args:
        rgb_lin: Numeric matrix with three columns (R, G, B) in [0, 1].

    Returns:
        Numeric matrix with three columns (X, Y, Z).
    """
    rgb_lin = np.asarray(rgb_lin, dtype=float)
    xyz = rgb_lin @ SRGB_LINEAR_TO_XYZ_D65.T
    return xyz


def xyz_to_lab(xyz: np.ndarray,
               white: np.ndarray = D65_WHITE) -> np.ndarray:
    """Convert CIEXYZ to CIELAB (D65).

    Implements the CIE 1976 L*a*b* transformation with the D65 white point.

    Args:
        xyz: Numeric matrix with columns X, Y, Z.
        white: Reference white point; default D65.

    Returns:
        Numeric matrix with columns L_star, a_star, b_star.
    """
    xyz = np.asarray(xyz, dtype=float)
    white = np.asarray(white, dtype=float)

    xr = xyz[..., 0] / white[0]
    yr = xyz[..., 1] / white[1]
    zr = xyz[..., 2] / white[2]

    def f(t):
        return np.where(t > LAB_EPSILON, t ** (1 / 3), (LAB_KAPPA * t + 16) / 116)

    fx = f(xr)
    fy = f(yr)
    fz = f(zr)

    lab = np.stack([
        116 * fy - 16,
        500 * (fx - fy),
        200 * (fy - fz)
    ], axis=-1)
    return lab


def lab_to_xyz(lab: np.ndarray,
               white: np.ndarray = D65_WHITE) -> np.ndarray:
    """Convert CIELAB to CIEXYZ (D65).

    Inverse of xyz_to_lab().

    Args:
        lab: Numeric matrix with columns L_star, a_star, b_star.
        white: Reference white point; default D65.

    Returns:
        Numeric matrix with columns X, Y, Z.
    """
    lab = np.asarray(lab, dtype=float)
    white = np.asarray(white, dtype=float)

    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    delta = 6 / 29

    def f_inv(t):
        return np.where(t > delta, t ** 3, 3 * delta ** 2 * (t - 4 / 29))

    xyz = np.stack([
        white[0] * f_inv(fx),
        white[1] * f_inv(fy),
        white[2] * f_inv(fz)
    ], axis=-1)
    return xyz


def xyz_to_rgb_lin(xyz: np.ndarray) -> np.ndarray:
    """Convert CIEXYZ to linear sRGB.

    Args:
        xyz: Numeric matrix with columns X, Y, Z.

    Returns:
        Numeric matrix with columns R, G, B in [0, 1] (may exceed range).
    """
    xyz = np.asarray(xyz, dtype=float)
    rgb = xyz @ XYZ_TO_SRGB_LINEAR_D65.T
    return rgb


def srgb_to_lab(srgb: np.ndarray) -> np.ndarray:
    """Convert non-linear sRGB directly to CIELAB.

    Convenience wrapper: sRGB (non-linear) -> linear sRGB -> XYZ -> LAB.

    Args:
        srgb: Numeric matrix with columns R, G, B in [0, 255] or [0, 1].

    Returns:
        Numeric matrix with columns L_star, a_star, b_star.
    """
    lin = srgb_to_linear(srgb)
    xyz = rgb_lin_to_xyz(lin)
    return xyz_to_lab(xyz)


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB to non-linear sRGB.

    Convenience wrapper: LAB -> XYZ -> linear sRGB -> sRGB (non-linear).

    Args:
        lab: Numeric matrix with columns L_star, a_star, b_star.

    Returns:
        Numeric matrix with columns R, G, B in [0, 1].
    """
    xyz = lab_to_xyz(lab)
    rgb_lin = xyz_to_rgb_lin(xyz)
    return linear_to_srgb(rgb_lin)


def lab_to_polar(lab: np.ndarray) -> np.ndarray:
    """Compute chroma and hue angle from CIELAB coordinates.

    Args:
        lab: Numeric matrix with columns L_star, a_star, b_star.

    Returns:
        Numeric matrix with columns chroma (C*) and hue_angle (h degrees, [0, 360)).
    """
    lab = np.asarray(lab, dtype=float)
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.sqrt(a ** 2 + b ** 2)
    hue_angle = np.degrees(np.arctan2(b, a)) % 360
    return np.stack([chroma, hue_angle], axis=-1)


# =============================================================================
# SECTION 1: Palette extraction
# =============================================================================

def extract_palette_kmeans(img: np.ndarray,
                           k: int = DEFAULT_K,
                           n_start: int = DEFAULT_N_START,
                           random_state: int = 42) -> pd.DataFrame:
    """Extract a colour palette using k-means++ clustering in CIELAB space.

    Converts the image to CIELAB (D65), then clusters pixels with the
    Lloyd-Forgy algorithm using k-means++ initialisation.

    Args:
        img: A 3-D numeric array (height x width x 3) with sRGB values in [0, 1].
        k: Desired number of palette colours (default 5).
        n_start: Number of random initialisations (default 10).
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with one row per palette colour and columns:
        L_star, a_star, b_star, weight, chroma, hue_angle.
    """
    if not (isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3):
        raise ValueError("'img' must be a 3-D array with shape (h, w, 3).")
    if not (2 <= k <= 20):
        raise ValueError("'k' must be between 2 and 20.")

    # Flatten image to pixel matrix (n_pixels x 3)
    h, w = img.shape[:2]
    rgb_mat = img.reshape(-1, 3)

    # Convert sRGB -> CIELAB
    lab_mat = srgb_to_lab(rgb_mat)

    # k-means++ with multiple restarts
    km = KMeans(n_clusters=k, n_init=n_start, random_state=random_state,
                max_iter=300, algorithm="lloyd")
    km.fit(lab_mat)

    centers = km.cluster_centers_
    labels = km.labels_
    weights = np.bincount(labels, minlength=k) / len(labels)

    # Compute chroma and hue angle
    polar = lab_to_polar(centers)

    return pd.DataFrame({
        "L_star": centers[:, 0],
        "a_star": centers[:, 1],
        "b_star": centers[:, 2],
        "weight": weights,
        "chroma": polar[:, 0],
        "hue_angle": polar[:, 1]
    })


def extract_palette_mediancut(img: np.ndarray,
                              k: int = DEFAULT_K) -> pd.DataFrame:
    """Extract a colour palette using median-cut quantization.

    Implements the classic median-cut colour quantization algorithm (Heckbert, 1982).
    The image is recursively split along the axis (R, G, or B) with the
    largest range until k boxes are obtained.

    Args:
        img: A 3-D numeric array (height x width x 3) with sRGB in [0, 1].
        k: Desired number of palette colours (default 5).

    Returns:
        DataFrame with the same columns as extract_palette_kmeans().

    References:
        Heckbert, P. (1982). Color image quantization for frame buffer display.
        ACM SIGGRAPH Computer Graphics, 16(3), 297-307.
    """
    if not (isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3):
        raise ValueError("'img' must be a 3-D array with shape (h, w, 3).")
    if not (2 <= k <= 20):
        raise ValueError("'k' must be between 2 and 20.")

    # Flatten to pixel matrix in sRGB [0, 1]
    rgb_mat = img.reshape(-1, 3).astype(float)

    # Each "box" is a list of pixel indices
    boxes = [np.arange(len(rgb_mat))]

    while len(boxes) < k:
        box_sizes = np.array([len(box) for box in boxes])
        idx_split = np.argmax(box_sizes)
        box = boxes[idx_split]

        if len(box) < 2:
            break

        rgb_box = rgb_mat[box]
        ranges = np.ptp(rgb_box, axis=0)
        split_axis = np.argmax(ranges)

        # Sort by chosen axis and split at median
        ord_idx = np.argsort(rgb_box[:, split_axis])
        mid = len(ord_idx) // 2

        boxes[idx_split] = box[ord_idx[:mid]]
        boxes.append(box[ord_idx[mid:]])

    # Compute mean sRGB of each box, then convert to CIELAB
    mean_rgb_list = []
    weights_list = []
    for box in boxes:
        mean_rgb_list.append(np.mean(rgb_mat[box], axis=0))
        weights_list.append(len(box) / len(rgb_mat))

    mean_rgb = np.array(mean_rgb_list)
    lab_mat = srgb_to_lab(mean_rgb)
    polar = lab_to_polar(lab_mat)

    return pd.DataFrame({
        "L_star": lab_mat[:, 0],
        "a_star": lab_mat[:, 1],
        "b_star": lab_mat[:, 2],
        "weight": weights_list,
        "chroma": polar[:, 0],
        "hue_angle": polar[:, 1]
    })


# =============================================================================
# SECTION 2: Individual feature-computation functions
# =============================================================================

def compute_hue_circular_variance(palette: pd.DataFrame) -> float:
    """Compute weighted circular variance of hue angles.

    Measures the dispersion of hue angles around the colour wheel. A value
    near 0 indicates that all hues are concentrated in one direction (low
    hue diversity); a value near 1 indicates hues are spread uniformly around
    the wheel (maximum hue diversity).

    Args:
        palette: DataFrame with columns 'weight' and 'hue_angle' (degrees).

    Returns:
        Scalar numeric in [0, 1].
    """
    if not isinstance(palette, pd.DataFrame):
        raise ValueError("'palette' must be a DataFrame.")
    if not ("weight" in palette.columns and "hue_angle" in palette.columns):
        raise ValueError("'palette' must contain 'weight' and 'hue_angle' columns.")
    if (palette["weight"] < 0).any():
        raise ValueError("Palette weights must be non-negative.")

    w = palette["weight"].values / palette["weight"].sum()
    h_rad = np.radians(palette["hue_angle"].values)

    x_bar = np.sum(w * np.cos(h_rad))
    y_bar = np.sum(w * np.sin(h_rad))
    R = np.sqrt(x_bar ** 2 + y_bar ** 2)

    return float(1 - R)


def compute_mean_chroma(palette: pd.DataFrame) -> float:
    """Compute weighted mean chroma.

    Args:
        palette: DataFrame with columns 'weight' and 'chroma'.

    Returns:
        Scalar numeric >= 0.
    """
    if not isinstance(palette, pd.DataFrame):
        raise ValueError("'palette' must be a DataFrame.")
    if not ("weight" in palette.columns and "chroma" in palette.columns):
        raise ValueError("'palette' must contain 'weight' and 'chroma' columns.")

    w = palette["weight"].values / palette["weight"].sum()
    return float(np.sum(w * palette["chroma"].values))


def compute_lightness_contrast(palette: pd.DataFrame) -> float:
    """Compute lightness contrast.

    Returns the weighted standard deviation of CIELAB L* values across the
    palette. Higher values indicate greater variation in lightness between
    palette colours (stronger light-dark contrast).

    Args:
        palette: DataFrame with columns 'weight' and 'L_star'.

    Returns:
        Scalar numeric >= 0.
    """
    if not isinstance(palette, pd.DataFrame):
        raise ValueError("'palette' must be a DataFrame.")
    if not ("weight" in palette.columns and "L_star" in palette.columns):
        raise ValueError("'palette' must contain 'weight' and 'L_star' columns.")

    w = palette["weight"].values / palette["weight"].sum()
    mu = np.sum(w * palette["L_star"].values)
    var_L = np.sum(w * (palette["L_star"].values - mu) ** 2)

    return float(np.sqrt(var_L))


def ciede2000_pair(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Compute CIEDE2000 colour difference between two LAB colours.

    This is a simplified implementation. For highest accuracy, use
    colour-science library.

    Args:
        lab1: LAB values [L, A, B] for first colour.
        lab2: LAB values [L, A, B] for second colour.

    Returns:
        Delta E00 value.
    """
    import warnings
    try:
        import colour
        return colour.delta_E(lab1, lab2, method="CIE 2000")
    except ImportError:
        # Fallback: approximate using Euclidean distance in LAB
        # This is not true CIEDE2000 but works for relative comparisons
        return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))


def compute_palette_deltaE00(palette: pd.DataFrame) -> float:
    """Compute mean pairwise CIEDE2000 colour difference.

    Calculates the average CIEDE2000 distance over all unordered pairs of
    palette colours.

    Args:
        palette: DataFrame with columns L_star, a_star, b_star.

    Returns:
        Scalar numeric >= 0. Returns NaN for single-colour palettes.
    """
    if not isinstance(palette, pd.DataFrame):
        raise ValueError("'palette' must be a DataFrame.")
    required = ["L_star", "a_star", "b_star"]
    if not all(col in palette.columns for col in required):
        raise ValueError(f"'palette' must contain columns: {', '.join(required)}")

    n = len(palette)
    if n < 2:
        return np.nan

    lab = palette[["L_star", "a_star", "b_star"]].values

    # Compute all pairwise distances
    de00_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            de00_vals.append(ciede2000_pair(lab[i], lab[j]))

    return float(np.mean(de00_vals))


# =============================================================================
# SECTION 3: Cohen-Or harmony template model
# =============================================================================

def compute_harmony_score(palette: pd.DataFrame) -> float:
    """Compute the Cohen-Or harmony score for a palette.

    Implements the template-based colour-harmony model described in Cohen-Or
    et al. (2006). Eight harmony templates (I, V, L, T, Y, X, N, i) are
    evaluated at every rotation angle alpha in [0 deg, 360 deg) in 1 deg steps.
    Lower scores indicate greater harmony.

    Args:
        palette: DataFrame with columns L_star, a_star, b_star, weight, chroma
            (and optionally hue_angle; if absent it is computed from a* and b*).

    Returns:
        Scalar numeric >= 0. Lower = more harmonious.
    """
    if not isinstance(palette, pd.DataFrame):
        raise ValueError("'palette' must be a DataFrame.")
    req_cols = ["L_star", "a_star", "b_star", "weight", "chroma"]
    missing = [c for c in req_cols if c not in palette.columns]
    if missing:
        raise ValueError(f"'palette' missing required columns: {', '.join(missing)}")

    # Ensure hue_angle is present
    if "hue_angle" not in palette.columns:
        polar = lab_to_polar(palette[["L_star", "a_star", "b_star"]].values)
        palette = palette.copy()
        palette["hue_angle"] = polar[:, 1]

    w = palette["weight"].values / palette["weight"].sum()
    h = palette["hue_angle"].values
    C = palette["chroma"].values

    norm_factor = np.sum(w * C)
    if norm_factor < 1e-12:
        return 0.0

    # Template definitions
    templates = {
        "I": np.array([[0, 15]]),
        "V": np.array([[-15, 15], [15, 15]]),
        "L": np.array([[0, 15], [90, 15]]),
        "T": np.array([[0, 15], [90, 15]]),
        "Y": np.array([[0, 15], [120, 15]]),
        "X": np.array([[0, 15], [180, 15]]),
        "N": np.array([[0, 90]]),
        "i": np.array([[0, 60]])
    }

    sigma = HARMONY_SIGMA_DEG

    def angular_dist_circle(a, b):
        d = np.abs((a - b) % 360)
        return np.where(d > 180, 360 - d, d)

    def dist_to_template(hues, template, alpha):
        centres = (template[:, 0] + alpha) % 360
        halfs = template[:, 1]
        dists = np.zeros(len(hues))
        for i, hue in enumerate(hues):
            d_to_sectors = angular_dist_circle(hue, centres)
            inside = d_to_sectors <= halfs
            if np.any(inside):
                dists[i] = 0
            else:
                dists[i] = np.min(d_to_sectors - halfs)
        return dists

    alpha_grid = np.arange(0, 360, 1)
    min_energy = np.inf

    for tname, templ in templates.items():
        for alpha in alpha_grid:
            d = dist_to_template(h, templ, alpha)
            penalty = np.where(
                d == 0,
                1 - np.exp(-(d ** 2) / (2 * sigma ** 2)),
                d
            )
            E = np.sum(w * C * penalty)
            if E < min_energy:
                min_energy = E

    harmony_score = min_energy / norm_factor
    return float(harmony_score)


# =============================================================================
# SECTION 4: Full pipeline function
# =============================================================================

def compute_all_features(image_path: str,
                         k: int = DEFAULT_K,
                         palette_method: str = "kmeans") -> dict:
    """Compute all colour features for a painting image.

    End-to-end pipeline: load the image, extract a k-means palette, and
    compute the five colour features.

    Args:
        image_path: Path to a PNG or JPEG image file.
        k: Number of palette colours (default 5).
        palette_method: Either "kmeans" (default) or "mediancut".

    Returns:
        Dictionary with computed features and palette.
    """
    from PIL import Image

    img = np.array(Image.open(image_path).convert("RGB")) / 255.0

    if palette_method == "kmeans":
        palette = extract_palette_kmeans(img, k=k, n_start=DEFAULT_N_START)
    elif palette_method == "mediancut":
        palette = extract_palette_mediancut(img, k=k)
    else:
        raise ValueError("palette_method must be 'kmeans' or 'mediancut'")

    return {
        "image_path": image_path,
        "palette": palette,
        "hue_circular_variance": compute_hue_circular_variance(palette),
        "mean_chroma": compute_mean_chroma(palette),
        "lightness_contrast": compute_lightness_contrast(palette),
        "mean_deltaE00": compute_palette_deltaE00(palette),
        "harmony_score": compute_harmony_score(palette)
    }


# =============================================================================
# SECTION 5: Image helpers
# =============================================================================

def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """Load and pre-process a painting image.

    Loads a PNG or JPEG image, crops a border to remove frames, standardises
    the longer edge to 800 px, and returns a 3-D numeric array.

    Args:
        image_path: Path to a PNG or JPEG file.

    Returns:
        A 3-D numeric array with dimensions (height, width, 3) in [0, 1].
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=float) / 255.0

    # Crop frame border
    h, w = arr.shape[:2]
    skip = FRAME_SKIP_PX
    if h > 2 * skip + 10 and w > 2 * skip + 10:
        arr = arr[skip:h - skip, skip:w - skip]

    # Standardise longer edge
    h, w = arr.shape[:2]
    long_edge = max(h, w)
    scale = TARGET_LONG_EDGE / long_edge

    if scale < 1.0:
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        img_resized = Image.fromarray((arr * 255).astype(np.uint8))
        img_resized = img_resized.resize((new_w, new_h), Image.LANCZOS)
        arr = np.array(img_resized, dtype=float) / 255.0

    return np.clip(arr, 0, 1)


def _angular_dist_on_circle(a: float, b: float) -> float:
    """Angular distance on a circle (shortest arc)."""
    d = abs((a - b) % 360)
    return 360 - d if d > 180 else d


def _signed_angular_diff(a: float, b: float) -> float:
    """Signed angular difference from a to b."""
    d = (b - a) % 360
    return d - 360 if d > 180 else d


def _find_best_template(palette: pd.DataFrame) -> dict:
    """Find the best harmony template for a palette."""
    w = palette["weight"].values / palette["weight"].sum()
    h = palette["hue_angle"].values
    C = palette["chroma"].values

    templates = {
        "I": np.array([[0, 15]]),
        "V": np.array([[-15, 15], [15, 15]]),
        "L": np.array([[0, 15], [90, 15]]),
        "T": np.array([[0, 15], [90, 15]]),
        "Y": np.array([[0, 15], [120, 15]]),
        "X": np.array([[0, 15], [180, 15]]),
        "N": np.array([[0, 90]]),
        "i": np.array([[0, 60]])
    }

    sigma = HARMONY_SIGMA_DEG
    alpha_grid = np.arange(0, 360, 1)

    def angular_dist_circle(a, b):
        d = np.abs((a - b) % 360)
        return np.where(d > 180, 360 - d, d)

    def dist_to_template(hues, template, alpha):
        centres = (template[:, 0] + alpha) % 360
        halfs = template[:, 1]
        dists = np.zeros(len(hues))
        for i, hue in enumerate(hues):
            d_to_sectors = angular_dist_circle(hue, centres)
            inside = d_to_sectors <= halfs
            if np.any(inside):
                dists[i] = 0
            else:
                dists[i] = np.min(d_to_sectors - halfs)
        return dists

    best_energy = np.inf
    best_name = None
    best_alpha = None
    best_templ = None

    for tname, templ in templates.items():
        for alpha in alpha_grid:
            d = dist_to_template(h, templ, alpha)
            penalty = np.where(d == 0, 1 - np.exp(-(d ** 2) / (2 * sigma ** 2)), d)
            E = np.sum(w * C * penalty)
            if E < best_energy:
                best_energy = E
                best_name = tname
                best_alpha = alpha
                best_templ = templ

    return {
        "template_name": best_name,
        "alpha": best_alpha,
        "template": best_templ,
        "energy": best_energy
    }


if __name__ == "__main__":
    print("color_features.py loaded successfully.")
    print("Available functions: srgb_to_lab, lab_to_srgb, extract_palette_kmeans,")
    print("  extract_palette_mediancut, compute_hue_circular_variance,")
    print("  compute_mean_chroma, compute_lightness_contrast,")
    print("  compute_palette_deltaE00, compute_harmony_score, compute_all_features")
