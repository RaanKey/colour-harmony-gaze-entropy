"""
config.py
Color Harmony Study -- Environment Setup and Configuration

This module provides color-space transformation matrices, global constants,
and reproducibility settings for the color feature computation pipeline.
"""

import numpy as np

# Reproducibility seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Package version tracking
PACKAGE_VERSIONS = {
    "numpy": "1.24.0+",
    "pandas": "2.0.0+",
    "scipy": "1.10.0+",
    "statsmodels": "0.14.0+",
    "matplotlib": "3.7.0+",
    "seaborn": "0.12.0+",
    "pymc": "5.0.0+",
    "pymer4": "0.8.0+",
    "pingouin": "0.5.0+",
    "scikit-learn": "1.3.0+",
}

# D65 white point (CIE 1931 2-degree observer)
D65_XN = 95.047
D65_YN = 100.000
D65_ZN = 108.883
D65_WHITE = np.array([D65_XN, D65_YN, D65_ZN])

# sRGB linearisation threshold
gamma_threshold = 0.04045

# sRGB (linear) -> CIEXYZ (D65) matrix
# Columns correspond to [R_linear, G_linear, B_linear]
# Rows correspond to [X, Y, Z]
SRGB_LINEAR_TO_XYZ_D65 = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])

# CIEXYZ -> sRGB (linear) inverse matrix
XYZ_TO_SRGB_LINEAR_D65 = np.linalg.inv(SRGB_LINEAR_TO_XYZ_D65)

# CIELAB 1976 parameters
LAB_KAPPA = 24389 / 27          # 903.296...
LAB_EPSILON = 216 / 24389       # 0.008856...

# Harmony-template parameters (Cohen-Or et al., 2006)
HARMONY_SIGMA_DEG = 15.0
HARMONY_SHIFT_DEG = 15.0
HARMONY_DELTA_L_MAX = 0.5

# Global constants for the palette pipeline
DEFAULT_K = 5
DEFAULT_N_START = 10
TARGET_LONG_EDGE = 800
FRAME_SKIP_PX = 20


# Sanity check: conversion round-trip
def _verify_round_trip():
    """Verify sRGB <-> XYZ round-trip consistency."""
    test_rgb = np.array([0.5, 0.3, 0.8])
    test_xyz = SRGB_LINEAR_TO_XYZ_D65 @ test_rgb
    test_rgb_back = XYZ_TO_SRGB_LINEAR_D65 @ test_xyz
    assert np.allclose(test_rgb, test_rgb_back, atol=1e-6), "sRGB<->XYZ round-trip failed"


_verify_round_trip()


# Print package info
if __name__ == "__main__":
    print("=== Color Harmony Pipeline -- Package versions ===")
    for pkg, version in sorted(PACKAGE_VERSIONS.items()):
        print(f"  {pkg}: {version}")
    print("===================================================")
    print("00_setup.py loaded successfully. All matrices consistent.")
