"""
power_analysis.py
Power Analysis for Linear Mixed-Effects Models

Study 1: Online Psychophysics (crossed random effects: participants x stimuli)
Study 2: Eye-Tracking (mediation: harmony -> fixation dispersion -> preference)

This script performs simulation-based power analysis for the key pre-registered
effects using Python equivalents of the R simr approach.

Usage:
    from power_analysis import run_power_analysis
    results = run_power_analysis()
"""

import os
import warnings
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import expit

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM

# Set plotting defaults
sns.set_theme(style="whitegrid", context="paper", palette="husl")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# 1. STUDY 1 -- DESIGN SPECIFICATION
# =============================================================================

# Design parameters
N_PARTICIPANTS = 180
N_EXPERT = 90
N_NOVICE = 90
N_STIMULI = 200
N_TRIALS = 72

# Correlation matrix among the 5 color features
FEATURE_CORR = np.array([
    [1.00, -0.30, 0.20, -0.10, 0.15],
    [-0.30, 1.00, -0.15, 0.05, -0.20],
    [0.20, -0.15, 1.00, 0.10, 0.25],
    [-0.10, 0.05, 0.10, 1.00, 0.05],
    [0.15, -0.20, 0.25, 0.05, 1.00]
])
FEATURE_NAMES = ["harmony", "circvar", "chroma", "lcontrast", "deltaE"]

# Fixed effects (standardized betas)
BETA = {
    "harmony": 0.15,
    "harmony_sq": -0.08,
    "circvar": 0.05,
    "circvar_sq": -0.06,
    "chroma": 0.12,
    "chroma_sq": -0.04,
    "lcontrast": 0.10,
    "lcontrast_sq": -0.05,
    "deltaE": 0.03,
    "deltaE_sq": -0.02,
    "expertise": 0.20,
    "expertise_harmony": -0.08,
    "expertise_circvar": 0.10,
    "expertise_chroma": -0.15,
    "expertise_lcontrast": 0.08,
    "expertise_deltaE": 0.02
}

# Variance components
SIGMA_P_SQ = 0.50  # Participant intercept variance
SIGMA_S_SQ = 0.30  # Stimulus intercept variance
SIGMA_E_SQ = 1.00  # Residual variance

ICC_PARTICIPANT = SIGMA_P_SQ / (SIGMA_P_SQ + SIGMA_S_SQ + SIGMA_E_SQ)
ICC_STIMULUS = SIGMA_S_SQ / (SIGMA_P_SQ + SIGMA_S_SQ + SIGMA_E_SQ)


def generate_correlated_features(n: int, corr_mat: np.ndarray) -> pd.DataFrame:
    """Generate correlated feature values using multivariate normal.

    Args:
        n: Number of samples.
        corr_mat: Correlation matrix.

    Returns:
        DataFrame with correlated features.
    """
    mu = np.zeros(corr_mat.shape[0])
    vals = np.random.multivariate_normal(mu, corr_mat, size=n)
    return pd.DataFrame(vals, columns=FEATURE_NAMES)


def build_study1_data(n_pp: int = N_PARTICIPANTS,
                      n_stim: int = N_STIMULI,
                      n_trials: int = N_TRIALS) -> pd.DataFrame:
    """Build Study 1 trial-level data structure for power simulation.

    Creates a crossed random-effects design with correlated predictors.

    Args:
        n_pp: Number of participants.
        n_stim: Number of stimuli.
        n_trials: Trials per participant.

    Returns:
        DataFrame with trial-level data.
    """
    expertise = np.array([0] * (n_pp // 2) + [1] * (n_pp - n_pp // 2))

    # Build trial data
    records = []
    for p in range(n_pp):
        np.random.seed(p)
        chosen_stim = np.random.choice(n_stim, min(n_trials, n_stim), replace=False)
        for s in chosen_stim:
            records.append({
                "participant_id": p,
                "stimulus_id": s,
                "expertise": expertise[p]
            })
    data = pd.DataFrame(records)

    # Generate stimulus features
    np.random.seed(RANDOM_SEED)
    stim_features = generate_correlated_features(n_stim, FEATURE_CORR)
    stim_features["stimulus_id"] = range(n_stim)

    data = data.merge(stim_features, on="stimulus_id", how="left")

    # Add quadratic and interaction terms
    for feat in FEATURE_NAMES:
        data[f"{feat}_sq"] = data[feat] ** 2
        data[f"expertise_{feat}"] = data["expertise"] * data[feat]

    return data


def simulate_preference(data: pd.DataFrame,
                        beta: Dict[str, float] = None) -> pd.DataFrame:
    """Simulate preference ratings from the model.

    Args:
        data: Trial-level DataFrame.
        beta: Coefficient dictionary.

    Returns:
        DataFrame with simulated preference column.
    """
    if beta is None:
        beta = BETA

    n = len(data)
    pp_ids = data["participant_id"].values
    stim_ids = data["stimulus_id"].values
    n_pp = pp_ids.max() + 1
    n_stim = stim_ids.max() + 1

    # Random effects
    u_0 = np.random.normal(0, np.sqrt(SIGMA_P_SQ), n_pp)
    w_0 = np.random.normal(0, np.sqrt(SIGMA_S_SQ), n_stim)
    epsilon = np.random.normal(0, np.sqrt(SIGMA_E_SQ), n)

    # Fixed effects
    eta = (beta.get("harmony", 0) * data["harmony"].values +
           beta.get("harmony_sq", 0) * data["harmony"].values ** 2 +
           beta.get("circvar", 0) * data["circvar"].values +
           beta.get("circvar_sq", 0) * data["circvar"].values ** 2 +
           beta.get("chroma", 0) * data["chroma"].values +
           beta.get("chroma_sq", 0) * data["chroma"].values ** 2 +
           beta.get("lcontrast", 0) * data["lcontrast"].values +
           beta.get("lcontrast_sq", 0) * data["lcontrast"].values ** 2 +
           beta.get("deltaE", 0) * data["deltaE"].values +
           beta.get("deltaE_sq", 0) * data["deltaE"].values ** 2 +
           beta.get("expertise", 0) * data["expertise"].values +
           beta.get("expertise_harmony", 0) * data["expertise"].values * data["harmony"].values +
           beta.get("expertise_chroma", 0) * data["expertise"].values * data["chroma"].values)

    preference = eta + u_0[pp_ids] + w_0[stim_ids] + epsilon
    data = data.copy()
    data["preference"] = preference
    return data


def fit_study1_model(data: pd.DataFrame) -> Any:
    """Fit Study 1 mixed-effects model.

    Args:
        data: Trial-level DataFrame with preference.

    Returns:
        Fitted model or None.
    """
    formula = (
        "preference ~ harmony + I(harmony**2) + circvar + I(circvar**2) "
        "+ chroma + I(chroma**2) + lcontrast + I(lcontrast**2) + deltaE + I(deltaE**2) "
        "+ expertise + expertise:harmony + expertise:chroma"
    )
    try:
        model = smf.mixedlm(formula=formula, data=data, groups=data["participant_id"]).fit(reml=False)
        return model
    except Exception:
        try:
            model = smf.ols(formula=formula + " + C(participant_id)", data=data).fit()
            return model
        except Exception:
            return None


def power_simulation_single(data: pd.DataFrame,
                            target_term: str,
                            n_sim: int = 500,
                            alpha: float = 0.05) -> Dict[str, float]:
    """Run power simulation for a single effect.

    Args:
        data: Base data structure.
        target_term: Coefficient name to test.
        n_sim: Number of simulations.
        alpha: Significance level.

    Returns:
        Dictionary with power estimate and CI.
    """
    sig_count = 0
    for i in range(n_sim):
        sim_data = simulate_preference(data)
        model = fit_study1_model(sim_data)
        if model is None:
            continue

        if target_term in model.pvalues and model.pvalues[target_term] < alpha:
            sig_count += 1

    power = sig_count / n_sim
    ci = stats.binom.interval(0.95, n_sim, power)

    return {
        "power": power,
        "ci_lower": ci[0] / n_sim,
        "ci_upper": ci[1] / n_sim,
        "n_sim": n_sim
    }


def run_study1_primary_power(n_sim: int = 500) -> pd.DataFrame:
    """Run primary power analysis for Study 1 key effects.

    Args:
        n_sim: Number of simulations per effect.

    Returns:
        DataFrame with power results.
    """
    print("=== PRIMARY POWER ANALYSIS ===\n")

    data = build_study1_data()

    key_effects = {
        "harmony": "harmony",
        "harmony_sq": "I(harmony ** 2)",
        "expertise_harmony": "expertise:harmony",
        "expertise_chroma": "expertise:chroma"
    }
    key_labels = {
        "harmony": "Harmony (linear)",
        "harmony_sq": "Harmony (quadratic)",
        "expertise_harmony": "Expertise x Harmony",
        "expertise_chroma": "Expertise x Chroma"
    }
    key_betas = {
        "harmony": 0.15,
        "harmony_sq": -0.08,
        "expertise_harmony": -0.08,
        "expertise_chroma": -0.15
    }

    results = []
    for eff_name, eff_term in key_effects.items():
        print(f"Testing: {key_labels[eff_name]} (beta = {key_betas[eff_name]:.2f})")

        # Run simplified power estimation using t-test on coefficient
        sig_count = 0
        for i in range(n_sim):
            sim_data = simulate_preference(data)
            model = fit_study1_model(sim_data)
            if model is None:
                continue
            pvals = model.pvalues
            if eff_name in pvals and pvals[eff_name] < 0.05:
                sig_count += 1

        pwr = sig_count / n_sim if n_sim > 0 else 0
        ci = stats.binom.interval(0.95, n_sim, pwr) if n_sim > 0 else (0, 0)

        results.append({
            "Effect": key_labels[eff_name],
            "Beta": key_betas[eff_name],
            "Power": round(pwr, 3),
            "CI_lower": round(ci[0] / n_sim, 3),
            "CI_upper": round(ci[1] / n_sim, 3),
            "Target_Met": "YES" if pwr >= 0.90 else "NO"
        })

        print(f"  Power = {pwr:.3f} [{ci[0]/n_sim:.3f}, {ci[1]/n_sim:.3f}] "
              f"-- Target >= .90: {'MET' if pwr >= 0.90 else 'NOT MET'}\n")

    return pd.DataFrame(results)


# =============================================================================
# 5. SENSITIVITY ANALYSIS -- POWER CURVES
# =============================================================================

def run_sensitivity_analysis(n_sim: int = 200) -> pd.DataFrame:
    """Run sensitivity analysis: vary effect sizes and sample sizes.

    Args:
        n_sim: Simulations per cell.

    Returns:
        DataFrame with sensitivity results.
    """
    print("=== SENSITIVITY ANALYSIS ===\n")

    effect_sizes = [0.05, 0.10, 0.15, 0.20]
    sample_sizes = [100, 120, 150, 180, 200]
    trials_per_pp = [60, 72, 80]

    results = []
    for n_pp in sample_sizes:
        for n_trial in trials_per_pp:
            for es in effect_sizes:
                print(f"  N = {n_pp}, Trials = {n_trial}, ES = {es:.2f}")

                beta_sens = BETA.copy()
                beta_sens["harmony"] = es

                data = build_study1_data(n_pp=n_pp, n_trials=n_trial)

                sig_count = 0
                for i in range(n_sim):
                    sim_data = simulate_preference(data, beta_sens)
                    model = fit_study1_model(sim_data)
                    if model is not None and "harmony" in model.pvalues:
                        if model.pvalues["harmony"] < 0.05:
                            sig_count += 1

                pwr = sig_count / n_sim
                ci = stats.binom.interval(0.95, n_sim, pwr)

                results.append({
                    "Effect": "Harmony (linear)",
                    "Effect_Size": es,
                    "N_Participants": n_pp,
                    "N_Trials": n_trial,
                    "Power": round(pwr, 3),
                    "CI_lower": round(ci[0] / n_sim, 3),
                    "CI_upper": round(ci[1] / n_sim, 3)
                })

    return pd.DataFrame(results)


def plot_power_curves(sensitivity_results: pd.DataFrame,
                      output_dir: str = "/mnt/agents/output/figures/") -> str:
    """Plot power curves from sensitivity analysis.

    Args:
        sensitivity_results: DataFrame from run_sensitivity_analysis().
        output_dir: Directory to save plot.

    Returns:
        Path to saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    effect_sizes = sorted(sensitivity_results["Effect_Size"].unique())
    sample_sizes = sorted(sensitivity_results["N_Participants"].unique())

    colors = plt.cm.viridis(np.linspace(0, 1, len(effect_sizes)))

    for i, es in enumerate(effect_sizes):
        es_data = sensitivity_results[
            (sensitivity_results["Effect_Size"] == es) &
            (sensitivity_results["N_Trials"] == 72)
        ].sort_values("N_Participants")

        ax.plot(es_data["N_Participants"], es_data["Power"],
                color=colors[i], linewidth=2, marker="o", markersize=6,
                label=f"beta = {es:.2f}")

    ax.axhline(y=0.90, color="red", linestyle="--", alpha=0.7, label="Power = 0.90")
    ax.axhline(y=0.80, color="orange", linestyle=":", alpha=0.7, label="Power = 0.80")
    ax.set_xlabel("Number of Participants")
    ax.set_ylabel("Statistical Power")
    ax.set_title("Study 1: Power Curves for Harmony Effect\n"
                 f"(Crossed RE: participant ICC = {ICC_PARTICIPANT:.2f}, "
                 f"stimulus ICC = {ICC_STIMULUS:.2f})")
    ax.set_ylim(0, 1)
    ax.set_xticks(sample_sizes)
    ax.legend(title="Effect Size", loc="lower right")
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "power_curves_study1.png")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Power curves saved to: {path}")
    return path


# =============================================================================
# 7. STUDY 2 -- MEDIATION POWER ANALYSIS
# =============================================================================

def run_study2_mediation_power(n_sim: int = 500) -> Dict[str, Any]:
    """Run Study 2 mediation power analysis.

    Args:
        n_sim: Number of simulations.

    Returns:
        Dictionary with mediation power results.
    """
    print("=== STUDY 2: MEDIATION POWER ANALYSIS ===\n")

    STUDY2_N = 45
    STUDY2_STIM = 60

    # Path A: harmony -> entropy (beta = 0.12)
    BETA_A = 0.12
    # Path B: entropy -> preference (beta = 0.15)
    BETA_B = 0.15

    sig_a = 0
    sig_b = 0
    sig_both = 0

    for i in range(n_sim):
        # Generate data
        np.random.seed(i)
        n = STUDY2_N * STUDY2_STIM
        participant_id = np.repeat(np.arange(STUDY2_N), STUDY2_STIM)
        harmony = np.random.normal(0, 1, n)

        # Path A: entropy ~ harmony
        u_pp_entropy = np.random.normal(0, 0.4, STUDY2_N)
        entropy = (BETA_A * harmony + u_pp_entropy[participant_id] +
                   np.random.normal(0, np.sqrt(0.70), n))

        # Path B: preference ~ harmony + entropy
        u_pp_pref = np.random.normal(0, 0.45, STUDY2_N)
        preference = (0.10 * harmony + BETA_B * entropy +
                      u_pp_pref[participant_id] +
                      np.random.normal(0, np.sqrt(0.80), n))

        data = pd.DataFrame({
            "participant_id": participant_id,
            "harmony": harmony,
            "entropy": entropy,
            "preference": preference
        })

        # Fit path A
        try:
            model_a = smf.ols("entropy ~ harmony + C(participant_id)", data=data).fit()
            p_a = model_a.pvalues.get("harmony", 1.0)
        except Exception:
            p_a = 1.0

        # Fit path B
        try:
            model_b = smf.ols("preference ~ harmony + entropy + C(participant_id)", data=data).fit()
            p_b = model_b.pvalues.get("entropy", 1.0)
        except Exception:
            p_b = 1.0

        if p_a < 0.05:
            sig_a += 1
        if p_b < 0.05:
            sig_b += 1
        if p_a < 0.05 and p_b < 0.05:
            sig_both += 1

    pwr_a = sig_a / n_sim
    pwr_b = sig_b / n_sim
    pwr_indirect = sig_both / n_sim

    ci_a = stats.binom.interval(0.95, n_sim, pwr_a)
    ci_b = stats.binom.interval(0.95, n_sim, pwr_b)

    print(f"Path a (harmony -> entropy, beta = 0.12):")
    print(f"  Power = {pwr_a:.3f} [{ci_a[0]/n_sim:.3f}, {ci_a[1]/n_sim:.3f}]")
    print(f"Path b (entropy -> preference, beta = 0.15):")
    print(f"  Power = {pwr_b:.3f} [{ci_b[0]/n_sim:.3f}, {ci_b[1]/n_sim:.3f}]")
    print(f"Joint indirect effect power (both paths significant):")
    print(f"  Power = {pwr_indirect:.3f} (Monte Carlo, {n_sim} simulations)")

    return {
        "pwr_path_a": pwr_a,
        "pwr_path_b": pwr_b,
        "pwr_indirect": pwr_indirect,
        "n_sim": n_sim
    }


# =============================================================================
# MASTER FUNCTION
# =============================================================================

def run_power_analysis(output_dir: str = "/mnt/agents/output/results/",
                       figure_dir: str = "/mnt/agents/output/figures/",
                       n_sim_primary: int = 500,
                       n_sim_sens: int = 200,
                       n_sim_study2: int = 500) -> Dict[str, Any]:
    """Run complete power analysis for both studies.

    Args:
        output_dir: Directory for saving results.
        figure_dir: Directory for saving figures.
        n_sim_primary: Simulations for primary power analysis.
        n_sim_sens: Simulations per cell for sensitivity analysis.
        n_sim_study2: Simulations for Study 2 mediation.

    Returns:
        Dictionary with all power analysis results.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("=" * 60)
    print("POWER ANALYSIS")
    print("=" * 60)
    print(f"\nStudy 1: Online Psychophysics (N = {N_PARTICIPANTS}, {N_TRIALS} trials/pp)")
    print(f"Study 2: Eye-Tracking (N = 45, 60 stimuli)")
    print(f"\nVariance Components:")
    print(f"  Participant variance: {SIGMA_P_SQ}")
    print(f"  Stimulus variance: {SIGMA_S_SQ}")
    print(f"  Residual variance: {SIGMA_E_SQ}")
    print(f"  ICC (participant): {ICC_PARTICIPANT:.3f}")
    print(f"  ICC (stimulus): {ICC_STIMULUS:.3f}")

    # 4. Primary Power Analysis
    primary_results = run_study1_primary_power(n_sim=n_sim_primary)

    # 5. Sensitivity Analysis
    sensitivity_results = run_sensitivity_analysis(n_sim=n_sim_sens)
    sensitivity_results.to_csv(os.path.join(output_dir, "power_sensitivity.csv"), index=False)
    print(f"Sensitivity table saved to: {os.path.join(output_dir, 'power_sensitivity.csv')}")

    # Plot power curves
    plot_power_curves(sensitivity_results, figure_dir)

    # 7. Study 2 Mediation Power
    study2_results = run_study2_mediation_power(n_sim=n_sim_study2)

    # Summary Table
    print("\n" + "=" * 60)
    print("POWER ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nSTUDY 1 -- Online Psychophysics (N = {N_PARTICIPANTS}, {N_TRIALS} trials/pp)")
    print("-" * 60)
    for _, row in primary_results.iterrows():
        print(f"  {row['Effect']:35s} beta = {row['Beta']:5.2f}  "
              f"Power = {row['Power']:.3f}  {row['Target_Met']}")

    print(f"\nSTUDY 2 -- Eye-Tracking (N = 45, 60 stimuli)")
    print("-" * 60)
    print(f"  Path a (harmony -> entropy)     beta =  0.12  Power = {study2_results['pwr_path_a']:.3f}")
    print(f"  Path b (entropy -> preference)  beta =  0.15  Power = {study2_results['pwr_path_b']:.3f}")
    print(f"  Joint indirect effect power:      Power = {study2_results['pwr_indirect']:.3f}")

    # Conclusion
    all_primary_met = all(primary_results["Target_Met"] == "YES")
    study2_met = study2_results["pwr_indirect"] >= 0.80

    print(f"\nAll Study 1 primary targets (>= .90): {'YES -- CONFIRMED' if all_primary_met else 'NO -- REVIEW NEEDED'}")
    print(f"Study 2 indirect effect target (>= .80): {'YES -- CONFIRMED' if study2_met else 'NO -- REVIEW NEEDED'}")

    return {
        "primary_power": primary_results,
        "sensitivity": sensitivity_results,
        "study2_mediation": study2_results
    }


if __name__ == "__main__":
    results = run_power_analysis()
    print("\nPower analysis complete.")
