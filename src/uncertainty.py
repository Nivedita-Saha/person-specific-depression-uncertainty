"""
Uncertainty quantification on the personalised model.

Two layers:
  1. MC dropout   - keep dropout active at inference; the spread across many
                    forward passes is a per-person epistemic uncertainty.
  2. Conformal    - calibrate an interval on the validation set so that test
                    intervals achieve a target coverage (e.g. 90%).

Also demonstrates an abstention rule: flag the most uncertain cases as
"refer to clinician" rather than committing to a score.

Run from the project root:
    python -m src.uncertainty
"""

from pathlib import Path
import numpy as np
import torch

from src.personalise import build_sequences, FEATURES
from src.model import GRURegressor, make_tensors, standardise

RESULTS_DIR = Path("results")
MODE = "personalised"
N_MC = 100           # number of stochastic forward passes
TARGET_COVERAGE = 0.90
ABSTAIN_FRACTION = 0.20   # flag the most uncertain 20% for clinician review


def enable_dropout(model):
    """Set only dropout layers to train mode so they stay active at inference."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_predict(model, X, n=N_MC):
    """Return (mean, std) over n stochastic forward passes."""
    model.eval()
    enable_dropout(model)
    preds = []
    with torch.no_grad():
        for _ in range(n):
            preds.append(model(X).numpy())
    preds = np.stack(preds)           # (n_passes, n_people)
    return preds.mean(0), preds.std(0)


def main():
    # Rebuild the exact same splits and standardisation used in training.
    sequences, info = build_sequences(personalise=True)
    X_train, _, _ = make_tensors(sequences, info, "train")
    X_val, y_val, _ = make_tensors(sequences, info, "val")
    X_test, y_test, test_rows = make_tensors(sequences, info, "test")
    (X_train, X_val, X_test), _ = standardise(X_train, X_val, X_test)

    # Load the trained personalised model.
    model = GRURegressor(n_features=len(FEATURES))
    model.load_state_dict(torch.load(RESULTS_DIR / f"model_{MODE}.pt"))

    # --- Conformal calibration on the validation set ---
    val_mean, _ = mc_predict(model, X_val)
    val_residuals = np.abs(y_val.numpy() - val_mean)
    # The (1 - alpha) empirical quantile of residuals is the half-width.
    q = np.quantile(val_residuals, TARGET_COVERAGE)
    print(f"Conformal half-width for {int(TARGET_COVERAGE*100)}% coverage: +/- {q:.2f} PHQ-8 points")

    # --- Apply to the test set ---
    test_mean, test_std = mc_predict(model, X_test)
    lower = test_mean - q
    upper = test_mean + q
    inside = (y_test.numpy() >= lower) & (y_test.numpy() <= upper)
    coverage = inside.mean()
    print(f"Empirical test coverage: {coverage:.0%} (target {int(TARGET_COVERAGE*100)}%)")
    print(f"Mean interval width: {2*q:.2f} PHQ-8 points")

    # --- Abstention: flag the most uncertain cases by MC-dropout std ---
    k = max(1, int(ABSTAIN_FRACTION * len(test_std)))
    abstain_idx = np.argsort(test_std)[::-1][:k]
    abstain_mask = np.zeros(len(test_std), dtype=bool)
    abstain_mask[abstain_idx] = True

    err = np.abs(y_test.numpy() - test_mean)
    mae_all = err.mean()
    mae_confident = err[~abstain_mask].mean()
    print(f"\nAbstention rule: flag most uncertain {int(ABSTAIN_FRACTION*100)}% "
          f"for clinician review.")
    print(f"MAE over all test cases:            {mae_all:.2f}")
    print(f"MAE over confident cases only:      {mae_confident:.2f}")
    print("If confident-only MAE is lower, the model 'knows when it doesn't know'.")

    # Save per-person uncertainty outputs for the write-up.
    np.savez(
        RESULTS_DIR / "uncertainty_personalised.npz",
        participant_id=test_rows["participant_id"].to_numpy(),
        y_true=y_test.numpy(),
        pred_mean=test_mean,
        pred_std=test_std,
        lower=lower,
        upper=upper,
        abstain=abstain_mask,
    )
    print("\nSaved per-person uncertainty to results/uncertainty_personalised.npz")


if __name__ == "__main__":
    main()