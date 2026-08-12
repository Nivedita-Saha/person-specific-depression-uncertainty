"""
Fairness slice: report performance separately by gender.

Real depression datasets (e.g. DAIC-WOZ) carry gender imbalance and documented
bias, so a single aggregate metric can hide unequal performance. Here we split
the personalised model's test predictions by gender and report error and
depression-accuracy per group.

Run from the project root:
    python -m src.fairness
"""

from pathlib import Path
import numpy as np

RESULTS_DIR = Path("results")
MODE = "personalised"


def metrics(y_true, pred):
    mae = np.abs(y_true - pred).mean()
    pred_label = (pred >= 10).astype(int)
    true_label = (y_true >= 10).astype(int)
    acc = (pred_label == true_label).mean()
    return mae, acc, len(y_true)


def main():
    data = np.load(RESULTS_DIR / f"test_{MODE}.npz", allow_pickle=True)
    y_true = data["y_test"]
    pred = data["pred"]
    gender = data["gender"].astype(str)

    print("=== Fairness slice by gender (personalised model, test set) ===")
    print(f"{'group':>8} {'n':>4} {'MAE':>7} {'dep-acc':>9}")

    overall_mae, overall_acc, n_all = metrics(y_true, pred)
    print(f"{'all':>8} {n_all:>4} {overall_mae:>7.2f} {overall_acc:>8.0%}")

    group_results = {}
    for g in ["F", "M"]:
        mask = gender == g
        if mask.sum() == 0:
            continue
        mae, acc, n = metrics(y_true[mask], pred[mask])
        group_results[g] = (mae, acc, n)
        print(f"{g:>8} {n:>4} {mae:>7.2f} {acc:>8.0%}")

    # Report the gap plainly.
    if "F" in group_results and "M" in group_results:
        mae_gap = abs(group_results["F"][0] - group_results["M"][0])
        acc_gap = abs(group_results["F"][1] - group_results["M"][1])
        print(f"\nMAE gap (|F - M|):            {mae_gap:.2f} PHQ-8 points")
        print(f"Depression-accuracy gap:     {acc_gap:.0%}")
        print("\nNote: with a small per-group test size these gaps are indicative,")
        print("not conclusive. On real data this slice is essential before any")
        print("claim of clinical usefulness.")


if __name__ == "__main__":
    main()