The Perceptual Architecture of Aesthetic Choice
How Colour Harmony Shapes Visual Preference Through Gaze Entropy
Under review at Nature Human Behaviour

Overview
This repository contains the analysis code and data for a two-study investigation of how computational colour harmony shapes aesthetic preference, and whether gaze entropy mediates this relationship.
•Study 1 (N = 180): Online psychophysics with preference and harmony ratings across four re-colouring conditions (+Harmony, Original, −Harmony, Filler).
•Study 2 (N = 45): Laboratory eye-tracking measuring gaze entropy during 4-second free viewing (32,549 fixations).
Key finding: Computational harmony scores positively predict aesthetic preference (r = 0.148) and gaze entropy (r = 0.251), but gaze entropy does not statistically mediate the harmony–preference link (indirect effect ab = −0.002, 95% CI [−0.005, 0.001]).

Repository Structure
.
|-- data/
|   |-- study1_data.csv              # Study 1 trial-level behavioural data (12,960 rows)
|   |-- study2_fixation_data.csv     # Study 2 fixation-level eye-tracking data (32,549 rows)
|   |-- study2_trial_data.csv        # Study 2 trial-level data (2,700 rows)
|
|-- analysis_study1.py               # Study 1: H1–H3 statistical analyses
|-- analysis_study2.py               # Study 2: gaze entropy, Bayesian mediation (H4)
|-- color_features.py                # Colour-space conversions & harmony score computation
|-- power_analysis.py                # Simulation-based power analysis
|-- config.py                        # Global constants (D65 white point, colour matrices)
|-- run_all.py                       # Master script: executes full analysis pipeline
|-- requirements.txt                 # Python dependencies
Data Description
study1_data.csv — 12,960 observations × 17 variables
Variable	Description
participant_id	Participant identifier (0–179)
stimulus_id	Palette identifier (0–199)
condition	Re-colouring condition (+Harmony, −Harmony, Original, Filler)
expertise	0 = novice, 1 = expert
harmony_score	Computational harmony score (raw)
circvar, chroma, lcontrast, deltaE00	Colour features (raw)
*_z	Z-scored colour features
preference	Aesthetic preference rating (0–100)
harmony_rating	Perceived harmony rating (0–100)
rt	Response time (ms)
study2_trial_data.csv — 2,700 observations × 5 variables
Variable	Description
participant_id	Participant identifier (0–44)
stimulus_id	Palette identifier (0–59)
harmony_score_z	Z-scored harmony score
entropy	Spatial gaze entropy (Shannon, 20×15 grid)
preference	Aesthetic preference rating (0–100)
expertise	0 = novice, 1 = expert
study2_fixation_data.csv — 32,549 fixations × 7 variables
Variable	Description
participant_id, stimulus_id	Trial identifiers
fixation_id	Fixation sequence number
x, y	Normalised fixation coordinates [0, 1]
duration	Fixation duration (ms)
start_time	Fixation onset (ms from trial start)

Installation
# Clone the repository
git clone https://github.com/RaanKey/colour-harmony-gaze-entropy.git
cd colour-harmony-gaze-entropy

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
Dependencies
•Python >= 3.10
•numpy, pandas, scipy, statsmodels
•matplotlib, seaborn
•PyMC >= 5.0 (Bayesian mediation analysis)
•scikit-learn

Usage
Run the full analysis pipeline
python run_all.py
Run individual analysis components
python analysis_study1.py   # Study 1: preference models, condition contrasts, expertise moderation
python analysis_study2.py   # Study 2: gaze entropy computation, Bayesian mediation
python power_analysis.py    # Power analysis for key effects
Quick start — reproduce H4 mediation
import pandas as pd
from scipy import stats

# Load Study 2 trial data
df = pd.read_csv('data/study2_trial_data.csv')

# Path a: harmony -> entropy
a, p_a = stats.pearsonr(df['harmony_score_z'], df['entropy'])
print(f"Path a: r = {a:.3f}, p = {p_a:.3e}")

# Path b: entropy -> preference
b, p_b = stats.pearsonr(df['entropy'], df['preference'])
print(f"Path b: r = {b:.3f}, p = {p_b:.3f}")

# Indirect effect
print(f"Indirect effect ab = {a * b:.3f}")

Reproducibility
All analyses are fully deterministic with random seed = 42. Results reported in the manuscript were produced with:
•Python 3.10+
•PyMC 5.10+ (Bayesian models)
•statsmodels 0.14+ (mixed-effects models)

Citation
If you use this code or data, please cite:
@article{wang2025colour,
  title={The Perceptual Architecture of Aesthetic Choice: How Colour Harmony Shapes Visual Preference Through Gaze Entropy},
  author={Wang, Haoran and [co-authors]},
  journal={Nature Human Behaviour},
  year={2025},
  note={Under review}
}

Ethics
This study was approved by the [University Name] Ethics Committee (Approval No. YJY-EC-2026-304). All participants provided written informed consent. The study was conducted in accordance with the Declaration of Helsinki.

License
This project is licensed under the MIT License — see the LICENSE file for details.
The data (data/*.csv) are made available for non-commercial research purposes. Please cite the paper if you use the data in your own research.

Contact
For questions about the code or data, please open an Issue or contact the corresponding author.
