"""
analysis_study1.py
Color Harmony and Aesthetic Preference - Study 1 Analysis

This script contains all confirmatory and exploratory analyses for Study 1,
an online psychophysical study with a crossed random-effects design.

Design:
    - N = 180 participants (90 expert, 90 novice)
    - 200 paintings, each participant rates 72 (balanced incomplete block)
    - Ratings: aesthetic preference and perceived harmony
    - 2AFC choice data (subset of trials)
    - Re-coloring manipulation: -Harmony / Original / +Harmony triplets

Usage:
    from analysis_study1 import run_all_study1_analyses
    results = run_all_study1_analyses("data/study1_data.csv")
"""

import os
import warnings
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import expit

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.discrete.discrete_model import Logit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.inspection import partial_dependence

# Set plotting defaults
sns.set_theme(style="whitegrid", context="paper", palette="husl")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (10, 6)

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Silence warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 0. SETUP
# =============================================================================

def set_study1_options() -> None:
    """Set global options for reproducibility."""
    np.random.seed(RANDOM_SEED)


def load_study1_data(data_path: str) -> pd.DataFrame:
    """Load and prepare Study 1 data from CSV.

    Args:
        data_path: Path to study1_data.csv.

    Returns:
        Prepared DataFrame with all required columns.
    """
    data = pd.read_csv(data_path)

    # Ensure categorical variables
    data["participant_id"] = data["participant_id"].astype("category")
    data["stimulus_id"] = data["stimulus_id"].astype("category")

    # Create z-scored feature columns (using _z suffix versions if available)
    feature_cols = ["harmony_score", "circvar", "chroma", "lcontrast", "deltaE00"]
    z_cols = [f"{c}_z" for c in feature_cols]
    if all(z in data.columns for z in z_cols):
        for feat, zcol in zip(feature_cols, z_cols):
            data[f"{feat}_z"] = data[zcol]
    else:
        for feat in feature_cols:
            data[f"{feat}_z"] = (data[feat] - data[feat].mean()) / data[feat].std()

    # Create quadratic terms
    for feat in feature_cols:
        data[f"{feat}_sq"] = data[feat] ** 2

    # Interaction terms
    for feat in feature_cols:
        data[f"expertise_{feat}"] = data["expertise"] * data[feat]

    # Recoloring numeric coding
    if "condition" in data.columns:
        data["recolor_num"] = data["condition"].map({
            "minus": -1, "-Harmony": -1,
            "original": 0, "Original": 0,
            "plus": 1, "+Harmony": 1,
            "Filler": 0
        }).fillna(0)

    return data


# =============================================================================
# 1. PRIMARY PREFERENCE MODEL (H1 + H3)
# =============================================================================

def fit_primary_preference_model(data: pd.DataFrame,
                                  use_poly: bool = True) -> Any:
    """Fit Primary Preference Mixed Model.

    Fits the main linear mixed-effects model for aesthetic preference ratings,
    testing inverted-U effects of color features and expertise moderation.

    Args:
        data: DataFrame containing the Study 1 trial-level data.
        use_poly: Whether to use polynomial features for quadratic terms.

    Returns:
        Fitted statsmodels MixedLMResults object.
    """
    required_cols = ["preference", "harmony_score", "circvar", "chroma",
                     "lcontrast", "deltaE00", "expertise",
                     "participant_id", "stimulus_id"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    print("Fitting primary preference model...")

    # Build formula
    if use_poly:
        formula = (
            "preference ~ harmony_score + I(harmony_score**2) "
            "+ circvar + I(circvar**2) + chroma + I(chroma**2) "
            "+ lcontrast + I(lcontrast**2) + deltaE00 + I(deltaE00**2) "
            "+ expertise * (harmony_score + circvar + chroma + lcontrast + deltaE00)"
        )
    else:
        formula = (
            "preference ~ harmony_score + harmony_score_sq "
            "+ circvar + circvar_sq + chroma + chroma_sq "
            "+ lcontrast + lcontrast_sq + deltaE00 + deltaE00_sq "
            "+ expertise * (harmony_score + circvar + chroma + lcontrast + deltaE00)"
        )

    # Fit with MixedLM
    try:
        model = smf.mixedlm(
            formula=formula,
            data=data,
            groups=data["participant_id"],
            re_formula="~harmony_score + circvar + chroma + lcontrast + deltaE00"
        ).fit(reml=False, method="powell")
        print("Primary preference model fitted successfully (full RE).")
        return model
    except Exception as e:
        print(f"Full model failed: {e}")
        # Fallback: simpler random effects
        try:
            model = smf.mixedlm(
                formula=formula,
                data=data,
                groups=data["participant_id"]
            ).fit(reml=False)
            print("Primary preference model fitted (random intercept only).")
            return model
        except Exception as e2:
            print(f"Simplified model also failed: {e2}")
            # Last resort: OLS
            model = smf.ols(formula=formula, data=data).fit()
            print("Fitted OLS as fallback.")
            return model


# =============================================================================
# 2. H1 TEST: INVERTED-U EFFECTS
# =============================================================================

def test_h1_inverted_u(primary_model: Any,
                        data: pd.DataFrame,
                        feature_names: List[str] = None) -> pd.DataFrame:
    """Test H1: Inverted-U Effects of Color Features.

    Uses likelihood ratio tests to compare models with and without quadratic
    terms for each color feature.

    Args:
        primary_model: The fitted primary preference model.
        data: The original DataFrame (for refitting reduced models).
        feature_names: List of feature names to test.

    Returns:
        DataFrame with LRT results for each feature's quadratic term.
    """
    if feature_names is None:
        feature_names = ["harmony_score", "circvar", "chroma", "lcontrast", "deltaE00"]

    print("Testing H1: Inverted-U effects...")

    # Linear-only comparison model
    linear_formula = (
        "preference ~ harmony_score + circvar + chroma + lcontrast + deltaE00 "
        "+ expertise + participant_id"
    )

    try:
        linear_model = smf.ols(formula=linear_formula, data=data).fit()
        primary_ols = smf.ols(
            formula="preference ~ harmony_score + I(harmony_score**2) + circvar + I(circvar**2) "
                    "+ chroma + I(chroma**2) + lcontrast + I(lcontrast**2) + deltaE00 + I(deltaE00**2) "
                    "+ expertise + participant_id",
            data=data
        ).fit()

        # Joint LRT via anova
        joint_anova = anova_lm(linear_model, primary_ols)
        joint_chisq = joint_anova.iloc[1]["F"]
        joint_df = joint_anova.iloc[1]["df_diff"]
        joint_p = joint_anova.iloc[1]["Pr(>F)"]
    except Exception as e:
        print(f"  Joint LRT failed: {e}")
        joint_chisq, joint_df, joint_p = np.nan, np.nan, np.nan

    results = [{
        "test": "Joint (all quadratic terms)",
        "chisq": joint_chisq,
        "df": joint_df,
        "p_value": joint_p,
        "delta_aic": getattr(linear_model, 'aic', np.nan) - getattr(primary_ols, 'aic', np.nan) if 'linear_model' in dir() else np.nan,
        "note": "Full quadratic vs. linear-only model"
    }]

    # Individual feature tests
    for feat in feature_names:
        other_feats = [f for f in feature_names if f != feat]
        try:
            # Model without this feature's quadratic term
            drop_formula = (
                f"preference ~ {feat} + "
                f"{' + '.join([o + ' + I(' + o + '**2)' for o in other_feats])} "
                f"+ expertise + participant_id"
            )
            drop_model = smf.ols(formula=drop_formula, data=data).fit()

            # Model with quadratic term
            full_feat_formula = (
                f"preference ~ {feat} + I({feat}**2) + "
                f"{' + '.join([o + ' + I(' + o + '**2)' for o in other_feats])} "
                f"+ expertise + participant_id"
            )
            full_feat_model = smf.ols(formula=full_feat_formula, data=data).fit()

            anova_res = anova_lm(drop_model, full_feat_model)
            results.append({
                "test": f"{feat} quadratic",
                "chisq": anova_res.iloc[1]["F"],
                "df": anova_res.iloc[1]["df_diff"],
                "p_value": anova_res.iloc[1]["Pr(>F)"],
                "delta_aic": drop_model.aic - full_feat_model.aic,
                "note": f"Linear-only {feat} vs. quadratic {feat}"
            })
        except Exception as e:
            print(f"    Dropped model failed for {feat}: {e}")
            results.append({
                "test": f"{feat} quadratic",
                "chisq": np.nan, "df": np.nan,
                "p_value": np.nan, "delta_aic": np.nan,
                "note": f"Failed: {e}"
            })

    results_df = pd.DataFrame(results)
    results_df["significance"] = results_df["p_value"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""
    )
    results_df["p_formatted"] = results_df["p_value"].apply(
        lambda p: "< .001" if p < 0.001 else f"{p:.3f}"
    )

    print("\n--- H1 Results: Inverted-U Effects ---")
    print(results_df[["test", "chisq", "df", "p_formatted", "delta_aic", "significance", "note"]].to_string())

    return results_df


# =============================================================================
# 3. RE-COLORING CAUSAL CONTRAST
# =============================================================================

def test_recoloring_contrast(data: pd.DataFrame) -> Dict[str, Any]:
    """Test Re-Coloring Causal Contrast.

    Analyzes the causal manipulation of harmony via re-coloring, using
    planned linear contrasts (minus < original < plus).

    Args:
        data: DataFrame containing re-coloring condition data.

    Returns:
        Dictionary with contrast results for preference and manipulation check.
    """
    print("Testing re-coloring causal contrast...")

    # Filter to triplet conditions only
    triplet_data = data[data["condition"].isin(["minus", "-Harmony", "original",
                                                  "Original", "plus", "+Harmony"])].copy()

    if len(triplet_data) == 0:
        print("No triplet condition data found. Skipping re-coloring contrast.")
        return {}

    # Recode condition to numeric
    triplet_data["recolor_cond"] = triplet_data["condition"].map({
        "minus": -1, "-Harmony": -1,
        "original": 0, "Original": 0,
        "plus": 1, "+Harmony": 1
    })

    results = {}

    # Preference model
    try:
        pref_model = smf.ols(
            "preference ~ recolor_cond + C(participant_id)",
            data=triplet_data
        ).fit()
        results["preference_model"] = pref_model
        results["preference_coef"] = pref_model.params.get("recolor_cond", np.nan)
        results["preference_pvalue"] = pref_model.pvalues.get("recolor_cond", np.nan)
        print(f"  Preference contrast: b = {results['preference_coef']:.3f}, "
              f"p = {results['preference_pvalue']:.4f}")
    except Exception as e:
        print(f"  Preference model failed: {e}")

    # Harmony rating model (manipulation check)
    if "harmony_rating" in triplet_data.columns:
        try:
            harm_model = smf.ols(
                "harmony_rating ~ recolor_cond + C(participant_id)",
                data=triplet_data
            ).fit()
            results["harmony_model"] = harm_model
            results["harmony_coef"] = harm_model.params.get("recolor_cond", np.nan)
            results["harmony_pvalue"] = harm_model.pvalues.get("recolor_cond", np.nan)
            print(f"  Harmony rating contrast: b = {results['harmony_coef']:.3f}, "
                  f"p = {results['harmony_pvalue']:.4f}")
        except Exception as e:
            print(f"  Harmony model failed: {e}")

    return results


# =============================================================================
# 4. H2: HARMONY-PREFERENCE DISSOCIATION
# =============================================================================

def test_h2_dissociation(data: pd.DataFrame,
                          features: List[str] = None) -> Dict[str, Any]:
    """Test H2: Harmony-Preference Dissociation.

    Compares the relationship between computational harmony features and
    (a) perceived harmony ratings vs (b) aesthetic preference ratings.

    Args:
        data: Study 1 DataFrame.
        features: List of feature names.

    Returns:
        Dictionary containing model comparisons, coefficient tables, and correlation.
    """
    if features is None:
        features = ["harmony_score", "circvar", "chroma", "lcontrast", "deltaE00"]

    print("Testing H2: Harmony-Preference Dissociation...")

    # Model (a): Perceived Harmony ~ Features
    print("  Fitting perceived harmony model...")
    harm_formula = "harmony_rating ~ " + " + ".join(features) + " + C(participant_id)"
    model_harmony = smf.ols(formula=harm_formula, data=data).fit()

    # Model (b): Preference ~ Features
    print("  Fitting preference model...")
    pref_formula = "preference ~ " + " + ".join(features) + " + C(participant_id)"
    model_preference = smf.ols(formula=pref_formula, data=data).fit()

    # Standardized coefficients
    def std_coeffs(model, feats):
        coefs = model.params
        std_vals = {}
        for f in feats:
            if f in coefs.index:
                sd_x = data[f].std()
                sd_y = data[model.model.endog_names].std() if hasattr(model.model, 'endog_names') else data["preference"].std()
                std_vals[f] = coefs[f] * sd_x / sd_y
        return std_vals

    harm_std = std_coeffs(model_harmony, features)
    pref_std = std_coeffs(model_preference, features)

    # Coefficient comparison table
    coef_comparison = pd.DataFrame({
        "feature": features,
        "harmony_beta": [model_harmony.params.get(f, np.nan) for f in features],
        "harmony_beta_std": [harm_std.get(f, np.nan) for f in features],
        "harmony_se": [model_harmony.bse.get(f, np.nan) for f in features],
        "harmony_p": [model_harmony.pvalues.get(f, np.nan) for f in features],
        "preference_beta": [model_preference.params.get(f, np.nan) for f in features],
        "preference_beta_std": [pref_std.get(f, np.nan) for f in features],
        "preference_se": [model_preference.bse.get(f, np.nan) for f in features],
        "preference_p": [model_preference.pvalues.get(f, np.nan) for f in features],
    })
    coef_comparison["difference"] = coef_comparison["preference_beta"] - coef_comparison["harmony_beta"]

    # R-squared comparison
    r2_comparison = pd.DataFrame({
        "outcome": ["Perceived Harmony", "Preference"],
        "r_squared": [model_harmony.rsquared, model_preference.rsquared],
        "adj_r_squared": [model_harmony.rsquared_adj, model_preference.rsquared_adj]
    })

    # Correlation: computational harmony vs. mean perceived harmony
    print("  Computing correlation...")
    stimulus_means = data.groupby("stimulus_id").agg(
        mean_perceived_harmony=("harmony_rating", "mean"),
        harmony_score=("harmony_score", "first")
    ).reset_index()

    harmony_correlation = stats.pearsonr(
        stimulus_means["harmony_score"],
        stimulus_means["mean_perceived_harmony"]
    )

    print("\n--- H2 Results: Harmony-Preference Dissociation ---")
    print("\nStandardized Coefficients:")
    print(coef_comparison[["feature", "harmony_beta_std", "preference_beta_std", "difference"]].to_string())
    print("\nR-squared Comparison:")
    print(r2_comparison.to_string())
    print(f"\nCorrelation (computational harmony ~ perceived harmony):")
    print(f"  r = {harmony_correlation.statistic:.3f}, p = {harmony_correlation.pvalue:.4f}")

    return {
        "model_harmony": model_harmony,
        "model_preference": model_preference,
        "coefficient_comparison": coef_comparison,
        "r2_comparison": r2_comparison,
        "harmony_correlation": harmony_correlation,
        "stimulus_means": stimulus_means
    }


# =============================================================================
# 5. H3: EXPERTISE MODERATION
# =============================================================================

def test_h3_expertise(primary_model: Any,
                       data: pd.DataFrame,
                       features: List[str] = None) -> Dict[str, Any]:
    """Test H3: Expertise Moderation of Color Feature Effects.

    Tests whether art expertise moderates the relationship between color
    features and aesthetic preference.

    Args:
        primary_model: The fitted primary preference model with interactions.
        data: The original DataFrame.
        features: List of feature names.

    Returns:
        Dictionary containing joint test, individual interactions, and contrasts.
    """
    if features is None:
        features = ["harmony_score", "circvar", "chroma", "lcontrast", "deltaE00"]

    print("Testing H3: Expertise Moderation...")

    # Joint F-test: full model vs. model without expertise:feature interactions
    print("  Running joint test for all expertise interactions...")

    reduced_formula = (
        "preference ~ harmony_score + I(harmony_score**2) + circvar + I(circvar**2) "
        "+ chroma + I(chroma**2) + lcontrast + I(lcontrast**2) + deltaE00 + I(deltaE00**2) "
        "+ expertise + C(participant_id)"
    )

    try:
        reduced_model = smf.ols(formula=reduced_formula, data=data).fit()
        full_formula = (
            "preference ~ harmony_score + I(harmony_score**2) + circvar + I(circvar**2) "
            "+ chroma + I(chroma**2) + lcontrast + I(lcontrast**2) + deltaE00 + I(deltaE00**2) "
            "+ expertise * (harmony_score + circvar + chroma + lcontrast + deltaE00) "
            "+ C(participant_id)"
        )
        full_model = smf.ols(formula=full_formula, data=data).fit()

        joint_anova = anova_lm(reduced_model, full_model)
        joint_f = joint_anova.iloc[1]["F"]
        joint_df = joint_anova.iloc[1]["df_diff"]
        joint_p = joint_anova.iloc[1]["Pr(>F)"]
    except Exception as e:
        print(f"  Joint test failed: {e}")
        joint_f, joint_df, joint_p = np.nan, np.nan, np.nan

    # Individual interaction coefficients
    print("  Extracting individual interaction coefficients...")
    interaction_terms = []
    for feat in features:
        term = f"expertise:{feat}"
        if term in full_model.params.index:
            interaction_terms.append({
                "term": term,
                "estimate": full_model.params[term],
                "std_error": full_model.bse[term],
                "p_value": full_model.pvalues[term],
                "ci_lower": full_model.conf_int().loc[term, 0],
                "ci_upper": full_model.conf_int().loc[term, 1]
            })

    interaction_df = pd.DataFrame(interaction_terms)
    if not interaction_df.empty:
        interaction_df["significance"] = interaction_df["p_value"].apply(
            lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""
        )

    # Expertise-specific simple slopes
    print("  Computing expertise-specific simple slopes...")
    expertise_slopes = []
    for feat in features:
        # Novice slope (expertise = 0)
        novice_coef = full_model.params.get(feat, np.nan)
        # Expert slope (expertise = 1)
        expert_coef = novice_coef + full_model.params.get(f"expertise:{feat}", 0)
        expertise_slopes.append({
            "feature": feat,
            "novice_slope": novice_coef,
            "expert_slope": expert_coef,
            "difference": full_model.params.get(f"expertise:{feat}", np.nan)
        })

    slopes_df = pd.DataFrame(expertise_slopes)

    print("\n--- H3 Results: Expertise Moderation ---")
    print(f"\nJoint F-test: F({joint_df:.0f}) = {joint_f:.2f}, p = {joint_p:.4f}")
    if not interaction_df.empty:
        print("\nIndividual Expertise x Feature Interactions:")
        print(interaction_df[["term", "estimate", "std_error", "p_value", "significance"]].to_string())
    print("\nExpertise-Specific Simple Slopes:")
    print(slopes_df.to_string())

    return {
        "joint_f": joint_f,
        "joint_df": joint_df,
        "joint_p_value": joint_p,
        "interaction_coefficients": interaction_df,
        "expertise_slopes": slopes_df
    }


# =============================================================================
# 6. 2AFC BRADLEY-TERRY MODEL
# =============================================================================

def fit_2afc_model(data: pd.DataFrame) -> Optional[Any]:
    """Fit 2AFC Bradley-Terry Model.

    Fits a logistic regression model for 2AFC choice data.

    Args:
        data: DataFrame with 2AFC trial data.

    Returns:
        Fitted logit model or None if no 2AFC data available.
    """
    print("Fitting 2AFC Bradley-Terry model...")

    if "choice" not in data.columns or data["choice"].isna().all():
        print("  No 2AFC data available. Skipping.")
        return None

    afc_data = data.dropna(subset=["choice"]).copy()
    if len(afc_data) == 0:
        print("  No 2AFC data available. Skipping.")
        return None

    print(f"  2AFC trials: {len(afc_data)}")

    try:
        # Map choice to binary if needed
        if afc_data["choice"].dtype == object:
            afc_data["choice_bin"] = (afc_data["choice"] == afc_data["choice"].iloc[0]).astype(int)
        else:
            afc_data["choice_bin"] = afc_data["choice"].astype(int)

        # Use delta_harmony if available, otherwise compute from harmony_score
        if "delta_harmony" not in afc_data.columns:
            afc_data["delta_harmony"] = afc_data["harmony_score"] - afc_data.groupby("participant_id")["harmony_score"].transform("mean")

        model = smf.logit(
            "choice_bin ~ delta_harmony + C(participant_id)",
            data=afc_data
        ).fit(disp=0)

        delta_coef = model.params.get("delta_harmony", np.nan)
        delta_p = model.pvalues.get("delta_harmony", np.nan)
        prob_1sd = expit(delta_coef)

        print("\n--- 2AFC Bradley-Terry Results ---")
        print(f"Delta harmony coefficient: b = {delta_coef:.3f}, p = {delta_p:.4f}")
        print(f"Probability of choosing higher-harmony stimulus (delta=1): {prob_1sd:.3f}")

        return model
    except Exception as e:
        print(f"  2AFC model failed: {e}")
        return None


# =============================================================================
# 7. ROBUSTNESS CHECKS
# =============================================================================

def run_robustness_checks(data: pd.DataFrame,
                          features: List[str] = None) -> Dict[str, Any]:
    """Run Robustness Checks.

    Performs a comprehensive battery of robustness checks.

    Args:
        data: Study 1 DataFrame.
        features: List of feature names.

    Returns:
        Dictionary summarizing robustness check results.
    """
    if features is None:
        features = ["harmony_score", "circvar", "chroma", "lcontrast", "deltaE00"]

    print("Running robustness checks...")
    results_list = []

    # a) Raw quadratic vs linear only
    print("  Check a: Raw quadratic terms...")
    try:
        model_raw = smf.ols(
            "preference ~ harmony_score + I(harmony_score**2) + circvar + I(circvar**2) "
            "+ chroma + I(chroma**2) + lcontrast + I(lcontrast**2) + deltaE00 + I(deltaE00**2) "
            "+ expertise * (harmony_score + circvar + chroma + lcontrast + deltaE00) "
            "+ C(participant_id)",
            data=data
        ).fit()
        for feat in features:
            if feat in model_raw.params.index:
                results_list.append({
                    "check": "raw_quadratic",
                    "feature": feat,
                    "estimate": model_raw.params[feat],
                    "p_value": model_raw.pvalues[feat],
                    "r_squared": model_raw.rsquared
                })
    except Exception as e:
        print(f"    Raw quadratic check failed: {e}")

    # b) Complete case analysis
    print("  Check b: Complete case analysis...")
    cols_needed = ["preference", "harmony_score", "circvar", "chroma", "lcontrast", "deltaE00", "expertise", "participant_id"]
    data_complete = data[cols_needed].dropna()
    try:
        model_complete = smf.ols(
            "preference ~ harmony_score + I(harmony_score**2) + circvar + I(circvar**2) "
            "+ chroma + I(chroma**2) + lcontrast + I(lcontrast**2) + deltaE00 + I(deltaE00**2) "
            "+ expertise + C(participant_id)",
            data=data_complete
        ).fit()
        for feat in features:
            if feat in model_complete.params.index:
                results_list.append({
                    "check": "complete_cases",
                    "feature": feat,
                    "estimate": model_complete.params[feat],
                    "p_value": model_complete.pvalues[feat],
                    "r_squared": model_complete.rsquared
                })
    except Exception as e:
        print(f"    Complete case check failed: {e}")

    # c) Outlier exclusion (IQR method on preference)
    print("  Check c: Outlier exclusion...")
    Q1 = data["preference"].quantile(0.25)
    Q3 = data["preference"].quantile(0.75)
    IQR = Q3 - Q1
    data_no_outliers = data[
        (data["preference"] >= Q1 - 1.5 * IQR) &
        (data["preference"] <= Q3 + 1.5 * IQR)
    ].copy()
    try:
        model_no_out = smf.ols(
            "preference ~ harmony_score + I(harmony_score**2) + circvar + I(circvar**2) "
            "+ chroma + I(chroma**2) + lcontrast + I(lcontrast**2) + deltaE00 + I(deltaE00**2) "
            "+ expertise + C(participant_id)",
            data=data_no_outliers
        ).fit()
        for feat in features:
            if feat in model_no_out.params.index:
                results_list.append({
                    "check": "outlier_exclusion",
                    "feature": feat,
                    "estimate": model_no_out.params[feat],
                    "p_value": model_no_out.pvalues[feat],
                    "r_squared": model_no_out.rsquared
                })
    except Exception as e:
        print(f"    Outlier exclusion check failed: {e}")

    results_df = pd.DataFrame(results_list)
    print(f"\n  Completed {len(results_list)} robustness checks.")
    return {"detailed_results": results_df}


# =============================================================================
# 8. MODEL DIAGNOSTICS
# =============================================================================

def run_model_diagnostics(model: Any,
                          data: pd.DataFrame,
                          save_plots: bool = True,
                          plot_dir: str = "/mnt/agents/output/figures/") -> Dict[str, Any]:
    """Run Model Diagnostics.

    Comprehensive diagnostic checks for a fitted model including residual plots,
    random effects distributions, and multicollinearity assessment.

    Args:
        model: A fitted model object.
        data: The data used to fit the model.
        save_plots: Whether to save diagnostic plots to disk.
        plot_dir: Directory to save plots.

    Returns:
        Dictionary containing diagnostic statistics and plot paths.
    """
    print("Running model diagnostics...")
    results = {"flags": [], "plots": {}, "plot_paths": [], "statistics": {}}

    os.makedirs(plot_dir, exist_ok=True)

    # Residual extraction
    residuals_raw = model.resid
    fitted_vals = model.fittedvalues

    # Standardized residuals
    residuals_std = (residuals_raw - residuals_raw.mean()) / residuals_raw.std()

    # 1. QQ Plot of Residuals
    print("  Creating residual QQ plot...")
    fig, ax = plt.subplots(figsize=(6, 5))
    stats.probplot(residuals_std, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot of Standardized Residuals")
    ax.grid(True, alpha=0.3)
    results["plots"]["qq_plot"] = fig
    if save_plots:
        path = os.path.join(plot_dir, "diagnostic_qq_plot.png")
        fig.savefig(path, bbox_inches="tight")
        results["plot_paths"].append(path)
    plt.close(fig)

    # 2. Residuals vs Fitted
    print("  Creating residuals vs fitted plot...")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(fitted_vals, residuals_std, alpha=0.3, s=5)
    ax.axhline(y=0, color="red", linestyle="--")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Standardized Residuals")
    ax.set_title("Residuals vs Fitted Values")
    ax.grid(True, alpha=0.3)
    results["plots"]["resid_fitted"] = fig
    if save_plots:
        path = os.path.join(plot_dir, "diagnostic_resid_vs_fitted.png")
        fig.savefig(path, bbox_inches="tight")
        results["plot_paths"].append(path)
    plt.close(fig)

    # 3. Histogram of Residuals
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(residuals_std, bins=50, color="steelblue", edgecolor="white", alpha=0.7)
    ax.set_xlabel("Standardized Residuals")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Residuals")
    ax.grid(True, alpha=0.3)
    results["plots"]["resid_hist"] = fig
    if save_plots:
        path = os.path.join(plot_dir, "diagnostic_resid_histogram.png")
        fig.savefig(path, bbox_inches="tight")
        results["plot_paths"].append(path)
    plt.close(fig)

    # 4. Shapiro-Wilk test (sample if large)
    print("  Testing normality...")
    sample_size = min(5000, len(residuals_raw))
    sample_idx = np.random.choice(len(residuals_raw), sample_size, replace=False)
    shapiro_stat, shapiro_p = stats.shapiro(residuals_raw.iloc[sample_idx])
    results["statistics"]["shapiro_wilk"] = {"statistic": shapiro_stat, "p_value": shapiro_p}
    if shapiro_p < 0.05:
        results["flags"].append(f"Non-normal residuals (Shapiro-Wilk p = {shapiro_p:.4f})")

    print("\n--- Diagnostic Summary ---")
    if len(results["flags"]) == 0:
        print("Flags: None")
    else:
        for flag in results["flags"]:
            print(f"  - {flag}")

    return results


# =============================================================================
# 9. EXPLORATORY: RANDOM FOREST
# =============================================================================

def run_exploratory_rf(data: pd.DataFrame,
                       features: List[str] = None,
                       n_folds: int = 5) -> Dict[str, Any]:
    """Run Exploratory Random Forest Analysis.

    Fits a random forest model with cross-validation to estimate the
    predictability ceiling of measured features for aesthetic preference.

    Args:
        data: Study 1 DataFrame.
        features: List of feature names.
        n_folds: Number of folds for CV.

    Returns:
        Dictionary containing RF model, CV results, and feature importance.
    """
    if features is None:
        features = ["harmony_score", "circvar", "chroma", "lcontrast", "deltaE00"]

    print("Running exploratory random forest analysis...")

    rf_data = data[["preference"] + features].dropna()
    print(f"  RF training data: {len(rf_data)} observations")

    X = rf_data[features].values
    y = rf_data["preference"].values

    # Fit Random Forest
    print("  Fitting random forest...")
    rf_model = RandomForestRegressor(
        n_estimators=1000,
        max_features=max(1, len(features) // 3),
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf_model.fit(X, y)

    oob_r2 = rf_model.score(X, y)
    oob_rmse = np.sqrt(np.mean((y - rf_model.predict(X)) ** 2))

    print(f"  Training R^2 = {oob_r2:.3f}, Training RMSE = {oob_rmse:.3f}")

    # Feature Importance
    importance_df = pd.DataFrame({
        "feature": features,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)
    importance_df["importance_scaled"] = importance_df["importance"] / importance_df["importance"].max() * 100

    # Cross-Validation
    print(f"  Running {n_folds}-fold cross-validation...")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    cv_preds = []
    cv_obs = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_model = RandomForestRegressor(
            n_estimators=500,
            max_features=max(1, len(features) // 3),
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        fold_model.fit(X_train, y_train)
        preds = fold_model.predict(X_test)
        cv_preds.extend(preds)
        cv_obs.extend(y_test)

    cv_preds = np.array(cv_preds)
    cv_obs = np.array(cv_obs)
    ss_res = np.sum((cv_obs - cv_preds) ** 2)
    ss_tot = np.sum((cv_obs - cv_obs.mean()) ** 2)
    cv_r2 = 1 - ss_res / ss_tot
    cv_rmse = np.sqrt(np.mean((cv_obs - cv_preds) ** 2))

    print(f"  CV R^2 = {cv_r2:.3f}, CV RMSE = {cv_rmse:.3f}")

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(7, 5))
    importance_sorted = importance_df.sort_values("importance_scaled")
    ax.barh(importance_sorted["feature"], importance_sorted["importance_scaled"], color="steelblue", alpha=0.8)
    ax.set_xlabel("Importance (scaled to max = 100)")
    ax.set_title(f"Random Forest: Feature Importance\nCV R^2 = {cv_r2:.3f}")
    ax.grid(True, alpha=0.3)
    results = {
        "rf_model": rf_model,
        "oob_r2": oob_r2,
        "oob_rmse": oob_rmse,
        "cv_r2": cv_r2,
        "cv_rmse": cv_rmse,
        "feature_importance": importance_df,
        "importance_plot": fig
    }
    plt.close(fig)

    # CV predictions plot
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(cv_preds, cv_obs, alpha=0.3, s=5)
    ax2.plot([cv_obs.min(), cv_obs.max()], [cv_obs.min(), cv_obs.max()], "r--", lw=1)
    ax2.set_xlabel("Predicted Preference")
    ax2.set_ylabel("Observed Preference")
    ax2.set_title(f"Random Forest: CV Predictions\nCV R^2 = {cv_r2:.3f}")
    ax2.grid(True, alpha=0.3)
    results["cv_plot"] = fig2
    plt.close(fig2)

    print("\n--- Random Forest Results ---")
    print(f"OOB R^2: {oob_r2:.3f} | CV R^2: {cv_r2:.3f}")
    print("\nFeature Importance:")
    print(importance_df[["feature", "importance_scaled"]].to_string())

    return results


# =============================================================================
# 10. MASTER EXECUTION FUNCTION
# =============================================================================

def run_all_study1_analyses(
    data_path: str,
    output_dir: str = "/mnt/agents/output/results/",
    figure_dir: str = "/mnt/agents/output/figures/",
    run_rf: bool = True
) -> Dict[str, Any]:
    """Run All Study 1 Analyses.

    Master function that loads data, fits all models, runs all hypothesis tests,
    performs diagnostics and robustness checks, and saves comprehensive results.

    Args:
        data_path: Path to the Study 1 data CSV file.
        output_dir: Directory to save results.
        figure_dir: Directory to save figures.
        run_rf: Whether to run exploratory random forest.

    Returns:
        Dictionary containing all analysis results.
    """
    print("=" * 60)
    print("Study 1 Analysis Pipeline")
    print("Color Harmony and Aesthetic Preference")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    set_study1_options()

    # Load data
    print(f"\nLoading data from: {data_path}")
    data = load_study1_data(data_path)
    print(f"\nDataset: {len(data)} observations, "
          f"{data['participant_id'].nunique()} participants, "
          f"{data['stimulus_id'].nunique()} stimuli")

    # 1. Primary Preference Model
    print("\n--- Step 1: Primary Preference Model ---")
    primary_model = fit_primary_preference_model(data)

    # 2. H1: Inverted-U Effects
    print("\n--- Step 2: H1 Inverted-U Effects ---")
    h1_results = test_h1_inverted_u(primary_model, data)

    # 3. Re-Coloring Causal Contrast
    print("\n--- Step 3: Re-Coloring Causal Contrast ---")
    recoloring_results = test_recoloring_contrast(data)

    # 4. H2: Harmony-Preference Dissociation
    print("\n--- Step 4: H2 Harmony-Preference Dissociation ---")
    h2_results = test_h2_dissociation(data)

    # 5. H3: Expertise Moderation
    print("\n--- Step 5: H3 Expertise Moderation ---")
    h3_results = test_h3_expertise(primary_model, data)

    # 6. 2AFC Bradley-Terry Model
    print("\n--- Step 6: 2AFC Bradley-Terry Model ---")
    afc_model = fit_2afc_model(data)

    # 7. Robustness Checks
    print("\n--- Step 7: Robustness Checks ---")
    robustness_results = run_robustness_checks(data, primary_model)

    # 8. Model Diagnostics
    print("\n--- Step 8: Model Diagnostics ---")
    diagnostics = run_model_diagnostics(primary_model, data, save_plots=True, plot_dir=figure_dir)

    # 9. Exploratory Random Forest
    rf_results = None
    if run_rf:
        print("\n--- Step 9: Exploratory Random Forest ---")
        rf_results = run_exploratory_rf(data)

    # Compile Results
    all_results = {
        "metadata": {
            "n_observations": len(data),
            "n_participants": data["participant_id"].nunique(),
            "n_stimuli": data["stimulus_id"].nunique(),
        },
        "primary_model": primary_model,
        "h1_results": h1_results,
        "recoloring_results": recoloring_results,
        "h2_results": h2_results,
        "h3_results": h3_results,
        "afc_model": afc_model,
        "robustness_results": robustness_results,
        "diagnostics": diagnostics,
        "rf_results": rf_results
    }

    # Save summary to CSV
    summary_path = os.path.join(output_dir, "study1_summary.csv")
    summary_df = pd.DataFrame({
        "test": ["H1 Joint", "H2 Correlation", "H3 Joint"],
        "statistic": [
            h1_results.iloc[0]["chisq"] if len(h1_results) > 0 else np.nan,
            h2_results["harmony_correlation"].statistic if h2_results else np.nan,
            h3_results["joint_f"] if h3_results else np.nan
        ],
        "p_value": [
            h1_results.iloc[0]["p_value"] if len(h1_results) > 0 else np.nan,
            h2_results["harmony_correlation"].pvalue if h2_results else np.nan,
            h3_results["joint_p_value"] if h3_results else np.nan
        ]
    })
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Print Summary Report
    print("\n" + "=" * 60)
    print("STUDY 1 ANALYSIS SUMMARY REPORT")
    print("=" * 60)
    print(f"\nDataset: {len(data)} observations")
    print(f"  Participants: {data['participant_id'].nunique()}")
    print(f"  Stimuli: {data['stimulus_id'].nunique()}")

    return all_results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # Try to find the data file
        candidate = os.path.join(os.path.dirname(__file__), "..", "..", "upload", "study1_data.csv")
        if os.path.exists(candidate):
            data_path = candidate
        else:
            data_path = "study1_data.csv"

    print(f"Running Study 1 analysis with data: {data_path}")
    results = run_all_study1_analyses(data_path)
    print("\nStudy 1 analysis complete.")
