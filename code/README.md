# Color Harmony Study -- Python Analysis Pipeline

Complete Python analysis pipeline for the study on color harmony in visual art.

## Project Structure

```
code/
    config.py                -- Color space constants and transformation matrices
    color_features.py        -- Color space conversions and feature computation
    power_analysis.py        -- Simulation-based power analysis
    analysis_study1.py       -- Study 1 behavioral data analysis
    analysis_study2.py       -- Study 2 eye-tracking analysis
    run_all.py               -- Master execution script
    requirements.txt         -- Python dependencies
    README.md                -- This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
python run_all.py
```

For a quick test run with reduced iterations:

```bash
python run_all.py --quick
```

### 3. Run Individual Analyses

```python
# Study 1
from analysis_study1 import run_all_study1_analyses
results = run_all_study1_analyses("data/study1_data.csv")

# Study 2
from analysis_study2 import run_all_study2_analyses
results = run_all_study2_analyses(
    "data/study2_trial_data.csv",
    "data/study2_fixation_data.csv"
)

# Power Analysis
from power_analysis import run_power_analysis
results = run_power_analysis()
```

## Data Files

The following real collected data files are required:

- `study1_data.csv` -- Study 1 behavioral data with columns:
  - `participant_id`, `stimulus_id`, `condition`, `expertise`
  - `harmony_score`, `circvar`, `chroma`, `lcontrast`, `deltaE00`
  - `harmony_score_z`, `circvar_z`, `chroma_z`, `lcontrast_z`, `deltaE00_z`
  - `preference`, `harmony_rating`, `rt`

- `study2_fixation_data.csv` -- Study 2 fixation data with columns:
  - `participant_id`, `stimulus_id`, `fixation_id`, `x`, `y`, `duration`, `start_time`

- `study2_trial_data.csv` -- Study 2 trial-level data with columns:
  - `participant_id`, `stimulus_id`, `harmony_score_z`, `entropy`, `preference`, `expertise`

## Module Descriptions

### config.py

Color space constants including:
- D65 white point values
- sRGB <-> CIEXYZ transformation matrices
- CIELAB 1976 parameters (kappa, epsilon)
- Cohen-Or harmony template parameters
- Palette extraction defaults

### color_features.py

Color feature computation pipeline:
- **Color space conversions**: sRGB <-> linear sRGB <-> CIEXYZ <-> CIELAB
- **Palette extraction**: k-means++ and median-cut algorithms in CIELAB space
- **Feature computation**: hue circular variance, mean chroma, lightness contrast, mean DeltaE00, harmony score
- **Harmony template model**: Cohen-Or et al. (2006) 8-template model with Gaussian fall-off

### analysis_study1.py

Study 1 confirmatory and exploratory analyses:
- **Primary preference model**: Mixed-effects regression with quadratic feature terms and expertise interactions
- **H1**: Inverted-U effects via likelihood ratio tests
- **H2**: Harmony-preference dissociation (correlation + coefficient comparison)
- **H3**: Expertise moderation (joint F-test + simple slopes)
- **Re-coloring contrast**: Planned linear contrast (-Harmony vs Original vs +Harmony)
- **2AFC Bradley-Terry**: Logistic regression for paired comparisons
- **Robustness checks**: Outlier exclusion, complete cases, polynomial type
- **Model diagnostics**: Residual plots, QQ plots, normality tests
- **Random forest**: Feature importance and cross-validated predictions

### analysis_study2.py

Study 2 eye-tracking analyses:
- **Gaze preprocessing**: I-VT fixation classification
- **Spatial entropy computation**: 2D histogram + Gaussian smoothing + Shannon entropy
- **Gaze metrics**: fixation SD, convex hull area, scanpath length, saccade amplitude
- **H4a**: Gaze-features model (entropy ~ color features)
- **H4 Bayesian mediation**: PyMC-based multilevel mediation (harmony -> entropy -> preference)
  - Falls back to bootstrap mediation if PyMC is unavailable
- **Temporal segmentation**: Windowed entropy analysis
- **AOI analysis**: Dwell time on high-chroma regions
- **Exploratory gaze measures**: Secondary gaze metrics as functions of features

### power_analysis.py

Simulation-based power analysis:
- **Study 1 primary power**: Key pre-registered effects (harmony linear, harmony quadratic, expertise x harmony, expertise x chroma)
- **Sensitivity analysis**: Power curves varying effect sizes (0.05-0.20), sample sizes (100-200), and trials per participant (60-80)
- **Study 2 mediation power**: Component-wise power for paths a and b, joint indirect effect

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.24.0 | Numerical computing, color space math |
| pandas | >=2.0.0 | Data manipulation |
| scipy | >=1.10.0 | Statistical tests, spatial computations |
| statsmodels | >=0.14.0 | Mixed-effects models, ANOVA, regression |
| matplotlib | >=3.7.0 | Plotting |
| seaborn | >=0.12.0 | Statistical visualization |
| pymc | >=5.0.0 | Bayesian mediation (optional) |
| scikit-learn | >=1.3.0 | k-means clustering, random forest |
| Pillow | >=9.0.0 | Image loading |

## Notes

- The Bayesian mediation in Study 2 will automatically fall back to bootstrap mediation if PyMC is not installed.
- Random seed is fixed to 42 throughout for reproducibility.
- All output is saved to `results/` (CSVs) and `figures/` (PNG plots).
