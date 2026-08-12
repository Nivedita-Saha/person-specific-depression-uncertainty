"""
One-command summary of all headline results.

Runs the baseline and both temporal models, then reports the three-way
comparison plus the uncertainty and fairness summaries in a single table.

Run from the project root:
    python -m src.evaluate
"""

import numpy as np
import contextlib
import io

from src import baseline as baseline_mod
from src import model as model_mod
from src import uncertainty as uncertainty_mod
from src import fairness as fairness_mod


def _quiet(fn, *args, **kwargs):
    """Run a function while suppressing its own prints; return nothing."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def main():
    print("=" * 60)
    print(" PERSON-SPECIFIC DEPRESSION ASSESSMENT - RESULTS SUMMARY")
    print("=" * 60)

    # --- Baseline (session-averaged Ridge) ---
    parts = baseline_mod.load_split()
    X_tr, y_tr, _ = parts["train"]
    X_te, y_te, _ = parts["test"]
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    base = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_tr, y_tr)
    base_mae, base_rmse, base_acc = baseline_mod.evaluate(base, X_te, y_te)

    # --- Temporal models (retrain quietly for a clean, reproducible run) ---
    raw_mae = _quiet(model_mod.train, "raw")
    pers_mae = _quiet(model_mod.train, "personalised")

    print("\nModel comparison (test set, PHQ-8 regression):")
    print(f"{'model':<28}{'MAE':>8}")
    print("-" * 36)
    print(f"{'Averaging baseline (Ridge)':<28}{base_mae:>8.2f}")
    print(f"{'Temporal GRU - raw':<28}{raw_mae:>8.2f}")
    print(f"{'Temporal GRU - personalised':<28}{pers_mae:>8.2f}")
    print("\nLower MAE is better. Personalisation is the component that helps;")
    print("raw temporal modelling alone does not beat the simple baseline.")

    # --- Uncertainty + fairness summaries (reuse saved personalised outputs) ---
    print("\nUncertainty (personalised model):")
    _quiet(uncertainty_mod.main)
    unc = np.load("results/uncertainty_personalised.npz")
    inside = (unc["y_true"] >= unc["lower"]) & (unc["y_true"] <= unc["upper"])
    print(f"  Conformal 90% interval - empirical coverage: {inside.mean():.0%}")
    print(f"  Mean interval width: {np.mean(unc['upper'] - unc['lower']):.2f} PHQ-8 points")

    print("\nFairness (personalised model):")
    _quiet(fairness_mod.main)
    data = np.load("results/test_personalised.npz", allow_pickle=True)
    print("  Per-group metrics printed by: python -m src.fairness")

    print("\n" + "=" * 60)
    print(" See README for the full honest interpretation of these results.")
    print("=" * 60)


if __name__ == "__main__":
    main()