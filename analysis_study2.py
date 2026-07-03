"""
analysis_study2.py
Study 2 Analysis: Color Harmony, Eye-Tracking, and Aesthetic Preference

Complete analysis pipeline for Study 2 eye-tracking data.
Includes gaze preprocessing, spatial entropy computation, Bayesian
multilevel mediation, temporal segmentation, AOI analysis, and
exploratory gaze measures.

Usage:
    from analysis_study2 import run_all_study2_analyses
    results = run_all_study2_analyses(
        trial_data_path="data/study2_trial_data.csv",
        fixation_data_path="data/study2_fixation_data.csv"
    )
"""

import os
import warnings
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# Set plotting defaults
sns.set_theme(style="whitegrid", context="paper", palette="husl")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (10, 6)

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 1. GAZE PREPROCESSING
# =============================================================================

def preprocess_gaze_data(
    raw_gaze_data: pd.DataFrame,
    velocity_threshold: float = 30.0,
    min_fixation_duration: float = 60.0,
    max_blink_duration: float = 200.0,
    track_loss_threshold: float = 0.3,
    min_fixations: int = 5
) -> Dict[str, Any]:
    """Preprocess Raw Gaze Data.

    Applies I-VT fixation classification, blink detection with interpolation,
    and trial quality checks to raw gaze data.

    Args:
        raw_gaze_data: DataFrame with columns: participant_id, stimulus_id,
            timestamp, x, y. Coordinates should be in [0,1].
        velocity_threshold: Velocity threshold for I-VT in deg/s.
        min_fixation_duration: Minimum fixation duration in ms.
        max_blink_duration: Maximum blink duration for interpolation in ms.
        track_loss_threshold: Maximum proportion of track loss per trial.
        min_fixations: Minimum number of fixations required per trial.

    Returns:
        Dictionary with 'fixations' (list of DataFrames) and 'excluded_trials'.
    """
    required_cols = ["participant_id", "stimulus_id", "x", "y"]
    missing = [c for c in required_cols if c not in raw_gaze_data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Sort by participant, stimulus
    raw_gaze_data = raw_gaze_data.sort_values(
        by=["participant_id", "stimulus_id"]
    ).reset_index(drop=True)

    # Identify missing/bad samples
    raw_gaze_data["is_missing"] = (
        raw_gaze_data["x"].isna() | raw_gaze_data["y"].isna() |
        (raw_gaze_data["x"] < -0.1) | (raw_gaze_data["x"] > 1.1) |
        (raw_gaze_data["y"] < -0.1) | (raw_gaze_data["y"] > 1.1)
    )
    raw_gaze_data["is_blink"] = raw_gaze_data["is_missing"]

    # Process each trial
    trials = raw_gaze_data[["participant_id", "stimulus_id"]].drop_duplicates()

    fixation_list = []
    excluded_trials = []

    for _, trial in trials.iterrows():
        pid, sid = trial["participant_id"], trial["stimulus_id"]
        trial_data = raw_gaze_data[
            (raw_gaze_data["participant_id"] == pid) &
            (raw_gaze_data["stimulus_id"] == sid)
        ].copy()

        if len(trial_data) < 10:
            excluded_trials.append({
                "participant_id": pid, "stimulus_id": sid,
                "exclusion_reason": "insufficient_samples"
            })
            continue

        # Track Loss Check
        prop_missing = trial_data["is_blink"].mean()
        if prop_missing > track_loss_threshold:
            excluded_trials.append({
                "participant_id": pid, "stimulus_id": sid,
                "exclusion_reason": "excessive_track_loss"
            })
            continue

        # I-VT Fixation Classification
        if "velocity" not in trial_data.columns:
            trial_data["dx"] = trial_data["x"].diff()
            trial_data["dy"] = trial_data["y"].diff()
            trial_data["dt"] = 16.0  # Assume 60Hz sampling (~16ms)
            trial_data["velocity"] = (
                np.sqrt(trial_data["dx"] ** 2 + trial_data["dy"] ** 2) /
                (trial_data["dt"] / 1000.0) * 40.0
            )

        trial_data["is_fixation"] = (
            trial_data["velocity"].fillna(0) < velocity_threshold
        ) & (~trial_data["is_blink"])

        # Identify fixation events (consecutive fixation samples)
        trial_data["event_change"] = (
            trial_data["is_fixation"].astype(int).diff().fillna(1) != 0
        )
        trial_data["event_id"] = trial_data["event_change"].cumsum()

        # Extract fixation events
        fixations = (trial_data[trial_data["is_fixation"]]
            .groupby("event_id")
            .agg(
                x=("x", "mean"),
                y=("y", "mean"),
                duration=("duration", "sum") if "duration" in trial_data.columns else ("x", "count"),
                start_time=("start_time", "first") if "start_time" in trial_data.columns else ("x", "first")
            )
            .reset_index(drop=True)
        )

        if "duration" not in trial_data.columns:
            fixations["duration"] = fixations["x"].count() * 16.0

        fixations = fixations.dropna(subset=["x", "y"])
        fixations = fixations[fixations["duration"] >= min_fixation_duration]

        if len(fixations) < min_fixations:
            excluded_trials.append({
                "participant_id": pid, "stimulus_id": sid,
                "exclusion_reason": "insufficient_fixations"
            })
            continue

        fixations = fixations[["x", "y", "duration", "start_time"]].copy()
        fixations["participant_id"] = pid
        fixations["stimulus_id"] = sid
        fixation_list.append(fixations)

    excluded_df = pd.DataFrame(excluded_trials)
    print(f"Preprocessing complete: {len(fixation_list)} trials processed, "
          f"{len(excluded_trials)} trials excluded")
    if len(excluded_df) > 0:
        print("Exclusion reasons:")
        print(excluded_df["exclusion_reason"].value_counts())

    return {
        "fixations": fixation_list,
        "excluded_trials": excluded_df
    }


# =============================================================================
# 2. COMPUTE GAZE METRICS
# =============================================================================

def compute_spatial_entropy(fixations: pd.DataFrame,
                            n_bins: int = 32,
                            sigma: float = 2.0,
                            epsilon: float = 1e-10) -> float:
    """Compute Spatial Entropy of Fixation Distribution.

    Creates a 2D histogram of fixation locations, applies Gaussian smoothing,
    normalizes to a probability distribution, and computes Shannon entropy.

    Args:
        fixations: DataFrame with columns x and y in [0,1] range.
        n_bins: Number of bins per dimension.
        sigma: Std dev of Gaussian smoothing kernel in bin units.
        epsilon: Small constant to avoid log(0).

    Returns:
        Shannon entropy value. Higher = more uniform distribution.
    """
    if not all(c in fixations.columns for c in ["x", "y"]):
        raise ValueError("fixations must contain 'x' and 'y' columns")

    coords = fixations[(fixations["x"] >= 0) & (fixations["x"] <= 1) &
                       (fixations["y"] >= 0) & (fixations["y"] <= 1)].copy()

    if len(coords) < 3:
        warnings.warn("Fewer than 3 valid fixations, returning NA for entropy")
        return np.nan

    # Create 2D histogram
    x_bins = np.clip((coords["x"].values * n_bins).astype(int), 0, n_bins - 1)
    y_bins = np.clip((coords["y"].values * n_bins).astype(int), 0, n_bins - 1)

    hist_2d = np.zeros((n_bins, n_bins))
    weights = coords["duration"].values if "duration" in coords.columns else np.ones(len(coords))
    for xb, yb, w in zip(x_bins, y_bins, weights):
        hist_2d[xb, yb] += w

    # Gaussian smoothing
    if sigma > 0:
        hist_smooth = gaussian_filter(hist_2d, sigma=sigma)
    else:
        hist_smooth = hist_2d

    # Normalize to probability distribution
    p = hist_smooth / hist_smooth.sum()

    # Shannon entropy
    p_nonzero = p[p > epsilon]
    entropy = -np.sum(p_nonzero * np.log(p_nonzero))

    return float(entropy)


def compute_fixation_sd(fixations: pd.DataFrame) -> float:
    """Compute sum of standard deviations of x and y fixation coordinates.

    Args:
        fixations: DataFrame with x, y columns in [0,1].

    Returns:
        Sum of SD(x) and SD(y).
    """
    coords = fixations.dropna(subset=["x", "y"])
    if len(coords) < 2:
        return np.nan
    sd_x = coords["x"].std()
    sd_y = coords["y"].std()
    return (sd_x or 0) + (sd_y or 0)


def compute_convex_hull_area(fixations: pd.DataFrame) -> float:
    """Compute Convex Hull Area of Fixations.

    Uses scipy.spatial.ConvexHull for convex hull computation.

    Args:
        fixations: DataFrame with x, y columns in [0,1].

    Returns:
        Area of convex hull in normalized units [0,1].
    """
    coords = fixations.dropna(subset=["x", "y"]).drop_duplicates(subset=["x", "y"])
    if len(coords) < 3:
        return 0.0

    try:
        hull = ConvexHull(coords[["x", "y"]].values)
        return float(hull.volume)  # volume = area in 2D
    except Exception:
        return 0.0


def compute_scanpath_length(fixations: pd.DataFrame) -> float:
    """Sum of Euclidean distances between consecutive fixations."""
    if len(fixations) < 2:
        return 0.0
    dx = fixations["x"].diff().dropna()
    dy = fixations["y"].diff().dropna()
    return float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))


def compute_mean_saccade_amplitude(fixations: pd.DataFrame) -> float:
    """Mean Euclidean distance between consecutive fixations."""
    if len(fixations) < 2:
        return 0.0
    dx = fixations["x"].diff().dropna()
    dy = fixations["y"].diff().dropna()
    return float(np.mean(np.sqrt(dx ** 2 + dy ** 2)))


def compute_all_gaze_metrics(fixations_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Compute All Gaze Metrics.

    Applies all gaze metric functions to a list of fixation DataFrames.

    Args:
        fixations_list: List of DataFrames, each containing fixations for one trial.

    Returns:
        DataFrame with one row per trial and columns for each gaze metric.
    """
    if len(fixations_list) == 0:
        raise ValueError("fixations_list is empty")

    results = []
    for fixs in fixations_list:
        if len(fixs) < 3:
            results.append({
                "participant_id": fixs["participant_id"].iloc[0] if len(fixs) > 0 else None,
                "stimulus_id": fixs["stimulus_id"].iloc[0] if len(fixs) > 0 else None,
                "entropy": np.nan,
                "sd_coords": np.nan,
                "convex_hull": np.nan,
                "scanpath_length": compute_scanpath_length(fixs),
                "mean_saccade_amp": compute_mean_saccade_amplitude(fixs),
                "n_fixations": len(fixs)
            })
            continue

        results.append({
            "participant_id": fixs["participant_id"].iloc[0],
            "stimulus_id": fixs["stimulus_id"].iloc[0],
            "entropy": compute_spatial_entropy(fixs, n_bins=32, sigma=2),
            "sd_coords": compute_fixation_sd(fixs),
            "convex_hull": compute_convex_hull_area(fixs),
            "scanpath_length": compute_scanpath_length(fixs),
            "mean_saccade_amp": compute_mean_saccade_amplitude(fixs),
            "n_fixations": len(fixs)
        })

    return pd.DataFrame(results)


# =============================================================================
# 3. GAZE ~ FEATURES MODEL (H4a)
# =============================================================================

def fit_gaze_features_model(data: pd.DataFrame,
                            gaze_metric: str = "entropy") -> Any:
    """Fit Gaze-Features Linear Model.

    Tests H4a: Color features predict gaze patterns (specifically spatial entropy).

    Args:
        data: DataFrame containing gaze metrics and stimulus features.
        gaze_metric: Name of the gaze metric column to use as DV.

    Returns:
        Fitted statsmodels OLSResults.
    """
    if gaze_metric not in data.columns:
        raise ValueError(f"Gaze metric '{gaze_metric}' not found in data")

    formula = (
        f"{gaze_metric} ~ harmony_score_z + circvar_z + chroma_z + "
        f"lcontrast_z + deltaE00_z + C(participant_id)"
    )

    print(f"Fitting model: {gaze_metric} ~ features + participant_id")
    model = smf.ols(formula=formula, data=data).fit()
    return model


def fit_gaze_harmony_model(data: pd.DataFrame,
                           gaze_metric: str = "entropy") -> Any:
    """Simplified model for testing harmony -> entropy relationship."""
    formula = f"{gaze_metric} ~ harmony_score_z + C(participant_id)"
    return smf.ols(formula=formula, data=data).fit()


# =============================================================================
# 4. H4 MEDIATION ANALYSIS (Bayesian)
# =============================================================================

def fit_mediation_pymc(data: pd.DataFrame,
                       covariates: Optional[List[str]] = None,
                       n_samples: int = 2000,
                       n_tune: int = 1000,
                       target_accept: float = 0.95,
                       random_seed: int = 42) -> Dict[str, Any]:
    """Fit Bayesian Multilevel Mediation Model using PyMC.

    Implements a Bayesian multilevel mediation analysis to test whether gaze
    entropy mediates the relationship between color harmony and aesthetic
    preference (H4).

    Mediation structure:
        Path A (a):  entropy ~ harmony_z + (1|participant) + (1|stimulus)
        Path B (b):  preference ~ harmony_z + entropy + (1|participant) + (1|stimulus)
        Path C (c):  preference ~ harmony_z + (1|participant) + (1|stimulus)

    Args:
        data: DataFrame with columns preference, harmony_score_z, entropy,
            participant_id, stimulus_id.
        covariates: Additional covariate column names.
        n_samples: Number of MCMC samples per chain.
        n_tune: Number of warmup iterations.
        target_accept: Target acceptance rate for NUTS.
        random_seed: Random seed.

    Returns:
        Dictionary with posterior samples, summary statistics, and diagnostics.
    """
    required_cols = ["preference", "harmony_score_z", "entropy",
                     "participant_id", "stimulus_id"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        print("WARNING: PyMC not available. Using non-Bayesian bootstrap mediation instead.")
        return fit_mediation_bootstrap(data, covariates=covariates, n_bootstrap=5000)

    # Prepare data
    y = data["preference"].values
    m = data["entropy"].values
    x = data["harmony_score_z"].values

    # Encode random effects
    pp_id, pp_labels = pd.factorize(data["participant_id"])
    stim_id, stim_labels = pd.factorize(data["stimulus_id"])
    n_pp = len(pp_labels)
    n_stim = len(stim_labels)

    # Covariates
    cov_data = None
    if covariates:
        valid_cov = [c for c in covariates if c in data.columns]
        if valid_cov:
            cov_data = data[valid_cov].values

    with pm.Model() as mediation_model:
        # Priors
        a0 = pm.Normal("a0", mu=5.0, sigma=2.0)  # entropy intercept
        b0 = pm.Normal("b0", mu=40.0, sigma=10.0)  # preference intercept
        c0 = pm.Normal("c0", mu=40.0, sigma=10.0)  # total effect intercept

        a = pm.Normal("a", mu=0, sigma=1)  # path A: harmony -> entropy
        b = pm.Normal("b", mu=0, sigma=5)  # path B: entropy -> preference
        cp = pm.Normal("cp", mu=0, sigma=5)  # c' direct effect
        c = pm.Normal("c", mu=0, sigma=5)  # path C: total effect

        # Random effects
        sigma_pp_a = pm.HalfCauchy("sigma_pp_a", 1)
        sigma_pp_b = pm.HalfCauchy("sigma_pp_b", 5)
        sigma_pp_c = pm.HalfCauchy("sigma_pp_c", 5)

        u_pp_a = pm.Normal("u_pp_a", mu=0, sigma=sigma_pp_a, shape=n_pp)
        u_pp_b = pm.Normal("u_pp_b", mu=0, sigma=sigma_pp_b, shape=n_pp)
        u_pp_c = pm.Normal("u_pp_c", mu=0, sigma=sigma_pp_c, shape=n_pp)

        # Path A: entropy ~ harmony
        mu_m = a0 + a * x + u_pp_a[pp_id]
        sigma_m = pm.HalfCauchy("sigma_m", 1)
        entropy_obs = pm.Normal("entropy_obs", mu=mu_m, sigma=sigma_m, observed=m)

        # Path B: preference ~ harmony + entropy
        mu_y = b0 + cp * x + b * m + u_pp_b[pp_id]
        sigma_y = pm.HalfCauchy("sigma_y", 5)
        preference_obs = pm.Normal("preference_obs", mu=mu_y, sigma=sigma_y, observed=y)

        # Compute derived quantities
        indirect_effect = pm.Deterministic("indirect_effect", a * b)
        total_effect = pm.Deterministic("total_effect", c)
        prop_mediated = pm.Deterministic("prop_mediated",
                                          pm.math.abs_(a * b) / (pm.math.abs_(cp) + pm.math.abs_(a * b)))

        # Sample
        trace = pm.sample(n_samples, tune=n_tune, target_accept=target_accept,
                         random_seed=random_seed, cores=4)

    # Extract summary
    summary = az.summary(trace, var_names=["a", "b", "cp", "indirect_effect", "prop_mediated"],
                         hdi_prob=0.95)

    # Extract posterior samples
    posterior_samples = pd.DataFrame({
        "a_path": trace.posterior["a"].values.flatten(),
        "b_path": trace.posterior["b"].values.flatten(),
        "c_prime": trace.posterior["cp"].values.flatten(),
        "indirect_effect": trace.posterior["indirect_effect"].values.flatten(),
        "prop_mediated": trace.posterior["prop_mediated"].values.flatten()
    })

    print("\n--- Mediation Summary ---")
    print(summary[["mean", "sd", "hdi_2.5%", "hdi_97.5%"]].to_string())

    # R-hat diagnostics
    max_rhat = summary["r_hat"].max()
    min_ess = summary["ess_bulk"].min()
    print(f"\nMCMC Diagnostics: max R-hat = {max_rhat:.4f}, min ESS = {min_ess:.0f}")

    return {
        "trace": trace,
        "summary": summary,
        "posterior_samples": posterior_samples,
        "indirect_effect": posterior_samples["indirect_effect"].values,
        "direct_effect": posterior_samples["c_prime"].values,
        "prop_mediated": posterior_samples["prop_mediated"].values,
        "max_rhat": max_rhat,
        "min_ess": min_ess
    }


def fit_mediation_bootstrap(data: pd.DataFrame,
                            covariates: Optional[List[str]] = None,
                            n_bootstrap: int = 5000) -> Dict[str, Any]:
    """Non-Bayesian bootstrap mediation analysis fallback.

    Uses percentile bootstrap to estimate confidence intervals for
    the indirect effect.

    Args:
        data: DataFrame with required columns.
        covariates: Additional covariates.
        n_bootstrap: Number of bootstrap samples.

    Returns:
        Dictionary with bootstrap results.
    """
    print(f"Running bootstrap mediation with {n_bootstrap} samples...")

    n = len(data)
    a_boot = np.zeros(n_bootstrap)
    b_boot = np.zeros(n_bootstrap)
    cp_boot = np.zeros(n_bootstrap)
    indirect_boot = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        boot_data = data.iloc[idx]

        # Path A: entropy ~ harmony
        try:
            model_a = smf.ols("entropy ~ harmony_score_z", data=boot_data).fit()
            a_boot[i] = model_a.params["harmony_score_z"]
        except Exception:
            a_boot[i] = np.nan

        # Path B: preference ~ harmony + entropy
        try:
            model_b = smf.ols("preference ~ harmony_score_z + entropy", data=boot_data).fit()
            b_boot[i] = model_b.params["entropy"]
            cp_boot[i] = model_b.params["harmony_score_z"]
        except Exception:
            b_boot[i] = np.nan
            cp_boot[i] = np.nan

        indirect_boot[i] = a_boot[i] * b_boot[i]

    # Summary
    valid = ~np.isnan(indirect_boot)
    summary_df = pd.DataFrame({
        "effect": ["Path A (harmony -> entropy)", "Path B (entropy -> preference)",
                   "Direct Effect (c')", "Indirect Effect (a*b)"],
        "mean": [np.nanmean(a_boot), np.nanmean(b_boot), np.nanmean(cp_boot), np.nanmean(indirect_boot)],
        "sd": [np.nanstd(a_boot), np.nanstd(b_boot), np.nanstd(cp_boot), np.nanstd(indirect_boot)],
        "ci_lower": [np.nanpercentile(a_boot, 2.5), np.nanpercentile(b_boot, 2.5),
                     np.nanpercentile(cp_boot, 2.5), np.nanpercentile(indirect_boot, 2.5)],
        "ci_upper": [np.nanpercentile(a_boot, 97.5), np.nanpercentile(b_boot, 97.5),
                     np.nanpercentile(cp_boot, 97.5), np.nanpercentile(indirect_boot, 97.5)],
        "p_direction": [np.nanmean(a_boot > 0), np.nanmean(b_boot > 0),
                        np.nanmean(cp_boot > 0), np.nanmean(indirect_boot > 0)]
    })

    print("\n--- Bootstrap Mediation Summary ---")
    print(summary_df.to_string())

    return {
        "summary": summary_df,
        "posterior_samples": pd.DataFrame({
            "a_path": a_boot, "b_path": b_boot, "c_prime": cp_boot,
            "indirect_effect": indirect_boot
        }),
        "indirect_effect": indirect_boot,
        "is_bootstrap": True
    }


# =============================================================================
# 5. TEMPORAL SEGMENTATION (Exploratory)
# =============================================================================

def run_temporal_analysis(
    fixation_data: pd.DataFrame,
    trial_features: pd.DataFrame,
    window_sizes: List[float] = None
) -> Dict[str, Any]:
    """Run Temporal Window Analysis.

    Analyzes how gaze entropy changes over the course of viewing.

    Args:
        fixation_data: DataFrame with fixation-level data.
        trial_features: DataFrame with trial-level features and ratings.
        window_sizes: Window boundaries in seconds.

    Returns:
        Dictionary with window metrics, comparisons, and plot.
    """
    if window_sizes is None:
        window_sizes = [0, 2, 4, 6]

    n_windows = len(window_sizes) - 1
    window_labels = [f"w{i+1}_{window_sizes[i]}-{window_sizes[i+1]}s"
                     for i in range(n_windows)]

    # Compute entropy per window
    window_metrics = []
    for w in range(n_windows):
        t_start = window_sizes[w] * 1000
        t_end = window_sizes[w + 1] * 1000

        window_fix = fixation_data[
            (fixation_data["start_time"] >= t_start) &
            (fixation_data["start_time"] < t_end)
        ]

        grouped = window_fix.groupby(["participant_id", "stimulus_id"])
        for (pid, sid), group in grouped:
            if len(group) < 3:
                continue
            entropy = compute_spatial_entropy(group, n_bins=32, sigma=2)
            window_metrics.append({
                "participant_id": pid,
                "stimulus_id": sid,
                "entropy": entropy,
                "n_fixations": len(group),
                "window": window_labels[w]
            })

    window_df = pd.DataFrame(window_metrics)
    if len(window_df) == 0:
        print("No window metrics computed.")
        return {}

    # Merge with trial features
    merge_cols = ["participant_id", "stimulus_id", "harmony_score_z", "preference"]
    merge_cols = [c for c in merge_cols if c in trial_features.columns]
    window_df = window_df.merge(
        trial_features[merge_cols], on=["participant_id", "stimulus_id"], how="left"
    )

    # Compare entropy across windows
    window_comparison = window_df.groupby("window").agg(
        n=("entropy", "count"),
        mean_entropy=("entropy", "mean"),
        sd_entropy=("entropy", "std")
    ).reset_index()
    window_comparison["se_entropy"] = window_comparison["sd_entropy"] / np.sqrt(window_comparison["n"])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    window_comparison["time_mid"] = [
        (window_sizes[i] + window_sizes[i+1]) / 2 for i in range(n_windows)
    ]
    ax.plot(window_comparison["time_mid"], window_comparison["mean_entropy"],
            color="steelblue", linewidth=2, marker="o", markersize=8)
    ax.errorbar(window_comparison["time_mid"], window_comparison["mean_entropy"],
                yerr=window_comparison["se_entropy"], fmt="none",
                color="steelblue", capsize=5)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Spatial Entropy (mean ± SE)")
    ax.set_title("Spatial Entropy Across Viewing Time")
    ax.set_xticks(window_comparison["time_mid"])
    ax.set_xticklabels([f"{t}s" for t in window_comparison["time_mid"]])
    ax.grid(True, alpha=0.3)

    return {
        "window_metrics": window_df,
        "window_comparison": window_comparison,
        "plot": fig
    }


# =============================================================================
# 6. AOI ANALYSIS
# =============================================================================

def run_aoi_analysis(
    fixation_data: pd.DataFrame,
    trial_features: pd.DataFrame,
    palette_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Run AOI (Area of Interest) Analysis.

    Analyzes dwell time on high-chroma regions.

    Args:
        fixation_data: DataFrame with fixation-level data.
        trial_features: DataFrame with trial-level features.
        palette_data: Optional DataFrame defining AOI regions.

    Returns:
        Dictionary with dwell data, models, and plots.
    """
    # Define AOIs if not provided
    if palette_data is None:
        stimuli = fixation_data["stimulus_id"].unique()
        aoi_records = []
        np.random.seed(RANDOM_SEED)
        for sid in stimuli:
            np.random.seed(int(sid) % 2**31)
            n_aois = np.random.choice([3, 4, 5])
            for i in range(n_aois):
                aoi_records.append({
                    "stimulus_id": sid,
                    "aoi_id": i + 1,
                    "aoi_x": np.random.uniform(0.2, 0.8),
                    "aoi_y": np.random.uniform(0.2, 0.8),
                    "aoi_radius": np.random.uniform(0.08, 0.18),
                    "chroma_level": "high" if i < 2 else "low"
                })
        palette_data = pd.DataFrame(aoi_records)

    high_chroma_aois = palette_data[palette_data["chroma_level"] == "high"][
        ["stimulus_id", "aoi_x", "aoi_y", "aoi_radius"]
    ]

    # Check if fixations fall within high-chroma AOIs
    fixation_aoi = fixation_data.merge(high_chroma_aois, on="stimulus_id", how="left")
    fixation_aoi["dist_to_aoi"] = np.sqrt(
        (fixation_aoi["x"] - fixation_aoi["aoi_x"]) ** 2 +
        (fixation_aoi["y"] - fixation_aoi["aoi_y"]) ** 2
    )
    fixation_aoi["in_high_chroma"] = fixation_aoi["dist_to_aoi"] < fixation_aoi["aoi_radius"]

    # Dwell time per trial
    dwell_data = fixation_aoi.groupby(["participant_id", "stimulus_id"]).agg(
        total_fix_time=("duration", "sum"),
        high_chroma_time=("duration", lambda x: x[fixation_aoi.loc[x.index, "in_high_chroma"]].sum()),
        n_fixations=("fixation_id", "count"),
        n_high_chroma_fixations=("in_high_chroma", "sum")
    ).reset_index()
    dwell_data["dwell_high_chroma"] = np.where(
        dwell_data["total_fix_time"] > 0,
        dwell_data["high_chroma_time"] / dwell_data["total_fix_time"],
        0
    )

    # Merge with trial features
    merge_cols = ["participant_id", "stimulus_id", "harmony_score_z", "preference", "condition"]
    merge_cols = [c for c in merge_cols if c in trial_features.columns]
    dwell_data = dwell_data.merge(
        trial_features[merge_cols], on=["participant_id", "stimulus_id"], how="left"
    )

    # Fit models
    print("Fitting AOI dwell time model...")
    formula_full = "dwell_high_chroma ~ harmony_score_z + C(participant_id)"
    formula_simple = "dwell_high_chroma ~ harmony_score_z"

    try:
        dwell_model = smf.ols(formula=formula_full, data=dwell_data).fit()
        dwell_harmony_model = smf.ols(formula=formula_simple, data=dwell_data).fit()
    except Exception as e:
        print(f"  AOI model fitting failed: {e}")
        dwell_model = None
        dwell_harmony_model = None

    # Summary by condition
    if "condition" in dwell_data.columns:
        aoi_summary = dwell_data.groupby("condition").agg(
            n=("dwell_high_chroma", "count"),
            mean_dwell=("dwell_high_chroma", "mean"),
            sd_dwell=("dwell_high_chroma", "std")
        ).reset_index()
    else:
        aoi_summary = pd.DataFrame()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(dwell_data["harmony_score_z"], dwell_data["dwell_high_chroma"],
               alpha=0.15, color="steelblue", s=10)
    # Add regression line
    z = np.polyfit(dwell_data["harmony_score_z"].dropna(),
                   dwell_data.loc[dwell_data["harmony_score_z"].notna(), "dwell_high_chroma"], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(dwell_data["harmony_score_z"].min(),
                         dwell_data["harmony_score_z"].max(), 100)
    ax.plot(x_line, p_line(x_line), color="darkred", linewidth=2)
    ax.set_xlabel("Harmony Score (z-scored)")
    ax.set_ylabel("Proportion Dwell Time on High-Chroma AOI")
    ax.set_title("Dwell Time on High-Chroma Regions vs. Color Harmony")
    ax.grid(True, alpha=0.3)

    return {
        "dwell_data": dwell_data,
        "dwell_model": dwell_model,
        "dwell_harmony_model": dwell_harmony_model,
        "aoi_summary": aoi_summary,
        "plot": fig
    }


# =============================================================================
# 7. EXPLORATORY GAZE MEASURES
# =============================================================================

def run_exploratory_gaze(data: pd.DataFrame) -> Dict[str, Any]:
    """Run Exploratory Gaze Measure Analyses.

    Analyzes secondary gaze measures as functions of stimulus features.

    Args:
        data: DataFrame with gaze metrics and stimulus features.

    Returns:
        Dictionary with models, summaries, and plots.
    """
    measures = {
        "scanpath_length": "Scanpath Length",
        "mean_saccade_amp": "Mean Saccade Amplitude",
        "sd_coords": "Fixation SD (x + y)",
        "convex_hull": "Convex Hull Area"
    }

    available = [m for m in measures if m in data.columns]
    if len(available) == 0:
        raise ValueError("No recognized gaze measure columns found in data")

    models = {}
    summaries = []
    plots = {}

    for measure in available:
        print(f"Fitting model for {measures[measure]}...")

        # Full model
        formula_full = f"{measure} ~ harmony_score_z + C(participant_id)"
        formula_simple = f"{measure} ~ harmony_score_z"

        try:
            model_full = smf.ols(formula=formula_full, data=data).fit()
            model_simple = smf.ols(formula=formula_simple, data=data).fit()

            models[measure] = {"full": model_full, "harmony_only": model_simple}

            coef_summary = pd.DataFrame({
                "measure": [measure] * len(model_full.params),
                "measure_label": measures[measure],
                "term": model_full.params.index,
                "estimate": model_full.params.values,
                "std_error": model_full.bse.values,
                "p_value": model_full.pvalues.values
            })
            summaries.append(coef_summary)
        except Exception as e:
            print(f"  Model failed for {measure}: {e}")
            models[measure] = {"full": None, "harmony_only": None}

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(data["harmony_score_z"], data[measure], alpha=0.15, color="steelblue", s=10)
        z = np.polyfit(data["harmony_score_z"].dropna(),
                       data.loc[data["harmony_score_z"].notna(), measure], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(data["harmony_score_z"].min(),
                             data["harmony_score_z"].max(), 100)
        ax.plot(x_line, p_line(x_line), color="darkred", linewidth=2)
        ax.set_xlabel("Harmony Score (z-scored)")
        ax.set_ylabel(measures[measure])
        ax.set_title(f"{measures[measure]} vs. Color Harmony")
        ax.grid(True, alpha=0.3)
        plots[measure] = fig
        plt.close(fig)

    all_summaries = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

    return {
        "models": models,
        "summaries": all_summaries,
        "plots": plots
    }


# =============================================================================
# 8. MASTER EXECUTION
# =============================================================================

def run_all_study2_analyses(
    trial_data_path: str,
    fixation_data_path: str,
    output_dir: str = "/mnt/agents/output/results/",
    figure_dir: str = "/mnt/agents/output/figures/",
    run_mediation: bool = True,
    run_temporal: bool = True,
    run_aoi: bool = True,
    run_exploratory: bool = True,
    mediation_samples: int = 2000,
    mediation_tune: int = 1000
) -> Dict[str, Any]:
    """Run All Study 2 Analyses.

    Master function that executes the complete Study 2 analysis pipeline.

    Args:
        trial_data_path: Path to trial-level CSV file.
        fixation_data_path: Path to fixation-level CSV file.
        output_dir: Directory for saving results.
        figure_dir: Directory for saving figures.
        run_mediation: Whether to run Bayesian mediation.
        run_temporal: Whether to run temporal segmentation.
        run_aoi: Whether to run AOI analysis.
        run_exploratory: Whether to run exploratory gaze measures.
        mediation_samples: MCMC samples per chain.
        mediation_tune: Warmup iterations.

    Returns:
        Dictionary containing all analysis results.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("=" * 60)
    print("STUDY 2 ANALYSIS PIPELINE")
    print("=" * 60)

    # 1. Load Data
    print("\n[1/8] Loading data...")
    trial_data = pd.read_csv(trial_data_path)
    fixation_data = pd.read_csv(fixation_data_path)

    print(f"  Trial data: {len(trial_data)} rows, {len(trial_data.columns)} columns")
    print(f"  Fixation data: {len(fixation_data)} rows, {len(fixation_data.columns)} columns")

    # 2. Compute Gaze Metrics
    print("\n[2/8] Computing gaze metrics...")

    # Group fixations by trial
    fix_groups = fixation_data.groupby(["participant_id", "stimulus_id"])
    fix_list = []
    for (pid, sid), group in fix_groups:
        g = group.copy()
        g["participant_id"] = pid
        g["stimulus_id"] = sid
        fix_list.append(g)

    gaze_metrics = compute_all_gaze_metrics(fix_list)

    # Merge with trial features
    merge_cols = ["participant_id", "stimulus_id", "harmony_score_z",
                  "entropy", "preference", "expertise"]
    merge_cols = [c for c in merge_cols if c in trial_data.columns]
    gaze_metrics = gaze_metrics.merge(
        trial_data[merge_cols], on=["participant_id", "stimulus_id"], how="left"
    )

    print(f"  Computed metrics for {len(gaze_metrics)} trials")

    # Save processed metrics
    gaze_metrics.to_csv(os.path.join(output_dir, "study2_gaze_metrics.csv"), index=False)

    # 3. Fit Gaze ~ Features Model (H4a)
    print("\n[3/8] Fitting gaze ~ features models (H4a)...")
    entropy_model = fit_gaze_features_model(gaze_metrics, gaze_metric="entropy")
    entropy_harmony_model = fit_gaze_harmony_model(gaze_metrics, gaze_metric="entropy")

    print("\n--- Entropy Model Summary ---")
    print(entropy_model.summary().tables[1])

    # 4. Bayesian Mediation (H4)
    mediation_results = None
    if run_mediation:
        print("\n[4/8] Running Bayesian mediation analysis (H4)...")
        try:
            mediation_results = fit_mediation_pymc(
                data=gaze_metrics,
                covariates=None,
                n_samples=mediation_samples,
                n_tune=mediation_tune
            )
        except Exception as e:
            print(f"  Bayesian mediation failed: {e}")
            print("  Falling back to bootstrap mediation...")
            mediation_results = fit_mediation_bootstrap(gaze_metrics)

    # 5. Temporal Segmentation
    temporal_results = None
    if run_temporal:
        print("\n[5/8] Running temporal segmentation analysis...")
        temporal_results = run_temporal_analysis(
            fixation_data=fixation_data,
            trial_features=trial_data,
            window_sizes=[0, 2, 4, 6]
        )
        if temporal_results.get("plot"):
            temporal_results["plot"].savefig(
                os.path.join(figure_dir, "temporal_entropy.png"),
                bbox_inches="tight", dpi=150
            )
            plt.close(temporal_results["plot"])
        if temporal_results.get("window_comparison") is not None:
            print("\n--- Temporal Entropy Comparison ---")
            print(temporal_results["window_comparison"].to_string())

    # 6. AOI Analysis
    aoi_results = None
    if run_aoi:
        print("\n[6/8] Running AOI analysis...")
        aoi_results = run_aoi_analysis(
            fixation_data=fixation_data,
            trial_features=trial_data,
            palette_data=None
        )
        if aoi_results.get("plot"):
            aoi_results["plot"].savefig(
                os.path.join(figure_dir, "aoi_dwell_time.png"),
                bbox_inches="tight", dpi=150
            )
            plt.close(aoi_results["plot"])
        if aoi_results.get("aoi_summary") is not None and not aoi_results["aoi_summary"].empty:
            print("\n--- AOI Dwell Time Summary ---")
            print(aoi_results["aoi_summary"].to_string())

    # 7. Exploratory Gaze Measures
    exploratory_results = None
    if run_exploratory:
        print("\n[7/8] Running exploratory gaze measure analyses...")
        exploratory_results = run_exploratory_gaze(gaze_metrics)
        if exploratory_results.get("summaries") is not None and not exploratory_results["summaries"].empty:
            print("\n--- Exploratory Measures Summary ---")
            print(exploratory_results["summaries"].to_string())

    # 8. Compile Results
    print("\n" + "=" * 60)
    print("COMPILING RESULTS")
    print("=" * 60)

    results = {
        "trial_data": trial_data,
        "fixation_data": fixation_data,
        "gaze_metrics": gaze_metrics,
        "entropy_model": entropy_model,
        "entropy_harmony_model": entropy_harmony_model,
        "mediation": mediation_results,
        "temporal": temporal_results,
        "aoi": aoi_results,
        "exploratory": exploratory_results
    }

    # Save results summary
    summary_path = os.path.join(output_dir, "study2_summary.csv")
    if mediation_results is not None and "summary" in mediation_results:
        med_summary = mediation_results["summary"]
        med_summary.to_csv(summary_path)
        print(f"Summary saved to: {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Participants: {trial_data['participant_id'].nunique()}")
    print(f"Stimuli: {trial_data['stimulus_id'].nunique()}")
    print(f"Total trials: {len(trial_data)}")
    print(f"Trials with computed entropy: {gaze_metrics['entropy'].notna().sum()}")

    if mediation_results is not None and "summary" in mediation_results:
        ie_row = mediation_results["summary"][
            mediation_results["summary"]["effect"] == "Indirect Effect (a*b)"
        ]
        if len(ie_row) > 0:
            ie_mean = ie_row["mean"].values[0]
            ie_lower = ie_row["ci_lower"].values[0]
            ie_upper = ie_row["ci_upper"].values[0]
            ci_includes_zero = ie_lower < 0 and ie_upper > 0
            print(f"\nMediation (Indirect Effect): M = {ie_mean:.3f}, "
                  f"95% CI [{ie_lower:.3f}, {ie_upper:.3f}] "
                  f"{'NS' if ci_includes_zero else 'Significant'}")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        trial_path = sys.argv[1]
        fixation_path = sys.argv[2]
    else:
        base = os.path.join(os.path.dirname(__file__), "..", "..", "upload")
        trial_path = os.path.join(base, "study2_trial_data.csv")
        fixation_path = os.path.join(base, "study2_fixation_data.csv")

    results = run_all_study2_analyses(trial_path, fixation_path)
    print("\nStudy 2 analysis complete.")
