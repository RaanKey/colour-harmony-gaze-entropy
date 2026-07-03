#!/usr/bin/env python3
"""
run_all.py
Master Execution Script
"Quantifying Color Harmony in Visual Art"

This script orchestrates the full analysis pipeline:
    1. Environment setup
    2. Color feature computation (utilities loaded)
    3. Power analysis
    4. Study 1 analyses (all confirmatory + robustness + exploratory)
    5. Study 2 analyses (eye-tracking + mediation + temporal)

Usage:
    python run_all.py                    # Full pipeline
    python run_all.py --quick            # Quick mode (reduced simulations)
    python run_all.py --skip-figures     # Skip figure generation

    Or from Python:
        from run_all import run_pipeline
        results = run_pipeline()
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, Any

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "..", "upload")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
CODE_DIR = SCRIPT_DIR

# Ensure output directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Add code directory to path
sys.path.insert(0, CODE_DIR)


class StepTimer:
    """Helper class to track timing of pipeline steps."""

    def __init__(self):
        self.steps = []
        self.start_time = None

    def start(self, name: str):
        """Start timing a step."""
        self.start_time = time.time()
        print(f"\n{'-' * 60}")
        print(f"[STEP {len(self.steps) + 1}/5] {name}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'-' * 60}")
        return self

    def end(self, status: str = "SUCCESS"):
        """End timing a step."""
        duration = time.time() - self.start_time
        name = f"Step {len(self.steps) + 1}"
        self.steps.append({
            "name": name,
            "duration": duration,
            "status": status
        })
        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"| Duration: {duration:.1f}s | Status: {status}")
        return duration


def print_banner():
    """Print the pipeline banner."""
    print("\n")
    print("=" * 60)
    print("  Quantifying Color Harmony in Visual Art")
    print("  Analysis Pipeline")
    print("=" * 60)
    print()


def run_step(timer: StepTimer, name: str, func, *args, **kwargs) -> Any:
    """Run a pipeline step with timing and error handling.

    Args:
        timer: StepTimer instance.
        name: Step name.
        func: Function to execute.
        *args, **kwargs: Arguments for func.

    Returns:
        Function result or None on failure.
    """
    timer.start(name)
    try:
        result = func(*args, **kwargs)
        timer.end("SUCCESS")
        return result
    except Exception as e:
        timer.end("FAILED")
        print(f"\n*** ERROR in step: {name} ***")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# STEP FUNCTIONS
# =============================================================================

def step1_setup() -> Dict[str, Any]:
    """Step 1: Environment Setup."""
    import config
    print("\nEnvironment setup complete.")
    print(f"Random seed: {config.RANDOM_SEED}")
    print(f"Package versions:")
    for pkg, version in sorted(config.PACKAGE_VERSIONS.items()):
        print(f"  {pkg}: {version}")
    return {"status": "ok"}


def step2_color_features() -> Dict[str, Any]:
    """Step 2: Color Feature Computation."""
    import color_features
    print("\nColor feature computation module loaded.")
    print("Available functions:")
    print("  - srgb_to_lab / lab_to_srgb")
    print("  - extract_palette_kmeans / extract_palette_mediancut")
    print("  - compute_hue_circular_variance / compute_mean_chroma")
    print("  - compute_lightness_contrast / compute_palette_deltaE00")
    print("  - compute_harmony_score / compute_all_features")
    return {"status": "ok"}


def step3_power_analysis(quick_mode: bool = False) -> Dict[str, Any]:
    """Step 3: Power Analysis."""
    from power_analysis import run_power_analysis

    n_sim_primary = 100 if quick_mode else 500
    n_sim_sens = 50 if quick_mode else 200
    n_sim_study2 = 100 if quick_mode else 500

    results = run_power_analysis(
        output_dir=RESULTS_DIR,
        figure_dir=FIGURES_DIR,
        n_sim_primary=n_sim_primary,
        n_sim_sens=n_sim_sens,
        n_sim_study2=n_sim_study2
    )
    return results


def step4_study1(quick_mode: bool = False) -> Dict[str, Any]:
    """Step 4: Study 1 Analyses."""
    from analysis_study1 import run_all_study1_analyses

    # Find data file
    data_path = os.path.join(DATA_DIR, "study1_data.csv")
    if not os.path.exists(data_path):
        # Try alternative paths
        candidates = [
            os.path.join(PROJECT_ROOT, "data", "study1_data.csv"),
            os.path.join(PROJECT_ROOT, "study1_data.csv"),
            "study1_data.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                data_path = c
                break

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Study 1 data not found at {data_path}")

    print(f"\nLoading Study 1 data from: {data_path}")
    results = run_all_study1_analyses(
        data_path=data_path,
        output_dir=RESULTS_DIR,
        figure_dir=FIGURES_DIR,
        run_rf=not quick_mode
    )
    return results


def step5_study2(quick_mode: bool = False) -> Dict[str, Any]:
    """Step 5: Study 2 Analyses."""
    from analysis_study2 import run_all_study2_analyses

    # Find data files
    trial_path = os.path.join(DATA_DIR, "study2_trial_data.csv")
    fixation_path = os.path.join(DATA_DIR, "study2_fixation_data.csv")

    for path in [trial_path, fixation_path]:
        if not os.path.exists(path):
            candidates = [
                os.path.join(PROJECT_ROOT, "data", os.path.basename(path)),
                os.path.join(PROJECT_ROOT, os.path.basename(path)),
                os.path.basename(path)
            ]
            for c in candidates:
                if os.path.exists(c):
                    if "trial" in path:
                        trial_path = c
                    else:
                        fixation_path = c
                    break

    if not os.path.exists(trial_path):
        raise FileNotFoundError(f"Study 2 trial data not found at {trial_path}")
    if not os.path.exists(fixation_path):
        raise FileNotFoundError(f"Study 2 fixation data not found at {fixation_path}")

    print(f"\nLoading Study 2 data from:")
    print(f"  Trial data: {trial_path}")
    print(f"  Fixation data: {fixation_path}")

    results = run_all_study2_analyses(
        trial_data_path=trial_path,
        fixation_data_path=fixation_path,
        output_dir=RESULTS_DIR,
        figure_dir=FIGURES_DIR,
        run_mediation=True,
        run_temporal=not quick_mode,
        run_aoi=True,
        run_exploratory=not quick_mode,
        mediation_samples=500 if quick_mode else 2000,
        mediation_tune=250 if quick_mode else 1000
    )
    return results


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(quick_mode: bool = False) -> Dict[str, Any]:
    """Run the complete analysis pipeline.

    Args:
        quick_mode: If True, use reduced iterations for faster execution.

    Returns:
        Dictionary with all step results.
    """
    print_banner()

    print(f"Mode: {'QUICK' if quick_mode else 'FULL'}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Code dir:     {CODE_DIR}")
    print(f"Data dir:     {DATA_DIR}")
    print(f"Results dir:  {RESULTS_DIR}")
    print(f"Figures dir:  {FIGURES_DIR}")

    timer = StepTimer()
    overall_start = time.time()

    # Run all steps
    results = {}
    results["setup"] = run_step(timer, "Environment Setup", step1_setup)
    results["color_features"] = run_step(timer, "Color Feature Computation", step2_color_features)
    results["power"] = run_step(timer, "Power Analysis", step3_power_analysis, quick_mode)
    results["study1"] = run_step(timer, "Study 1 Analyses", step4_study1, quick_mode)
    results["study2"] = run_step(timer, "Study 2 Analyses", step5_study2, quick_mode)

    overall_end = time.time()
    overall_duration = (overall_end - overall_start) / 60

    # Summary Report
    print("\n\n")
    print("=" * 60)
    print("               EXECUTION PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time: {overall_duration:.1f} minutes")
    print(f"  Steps completed: {sum(1 for s in timer.steps if s['status'] == 'SUCCESS')}/5")
    print(f"  Status: {'ALL SUCCESS' if all(s['status'] == 'SUCCESS' for s in timer.steps) else 'SOME FAILED'}")
    print("-" * 60)

    for step in timer.steps:
        symbol = "OK" if step["status"] == "SUCCESS" else "FAIL"
        print(f"  [{symbol}] {step['name']:30s}  {step['duration']:8.1f}s  ({step['status']})")

    print("=" * 60)

    # List output files
    print("\n--- Output Files ---")
    if os.path.exists(RESULTS_DIR):
        result_files = sorted(os.listdir(RESULTS_DIR))
        if result_files:
            print("\nResults:")
            for f in result_files:
                print(f"  {f}")

    if os.path.exists(FIGURES_DIR):
        figure_files = sorted(os.listdir(FIGURES_DIR))
        if figure_files:
            print("\nFigures:")
            for f in figure_files:
                print(f"  {f}")

    print("\n--- Reproducibility ---")
    print("To reproduce this analysis:")
    print("  1. Install Python >= 3.10 and required packages (see requirements.txt)")
    print("  2. Place data files in the data directory")
    print("  3. Run: python run_all.py")
    print("  4. Results will be saved to the results/ directory")
    print()

    return results


def main():
    """Parse command-line arguments and run pipeline."""
    parser = argparse.ArgumentParser(
        description="Color Harmony Study -- Full Analysis Pipeline"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode with reduced iterations"
    )
    parser.add_argument(
        "--skip-figures", action="store_true",
        help="Skip figure generation"
    )
    args = parser.parse_args()

    run_pipeline(quick_mode=args.quick)


if __name__ == "__main__":
    main()
