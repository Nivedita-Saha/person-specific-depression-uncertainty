"""
Explainability for the personalised model.

Attributes each PHQ-8 prediction back to input features using gradient x input
(saliency), summed over time per feature. Produces:
  - a global ranking of feature importance across the test set
  - a per-person plain-language explanation
  - a saved bar chart of global importance

Run from the project root:
    python -m src.explain
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.personalise import build_sequences, FEATURES
from src.model import GRURegressor, make_tensors, standardise

RESULTS_DIR = Path("results")
MODE = "personalised"

# Human-readable direction hints for the plain-language explanations.
READABLE = {
    "AU06_cheek_raiser": "cheek-raiser (smiling)",
    "AU12_lip_corner": "lip-corner pull (smiling)",
    "AU01_inner_brow": "inner-brow raise",
    "AU04_brow_lowerer": "brow-lowerer (frowning)",
    "gaze_x": "horizontal gaze",
    "gaze_y": "vertical gaze",
    "pose_Rx": "head pitch",
    "pose_Ry": "head yaw",
}


def attributions(model, X):
    """Return per-feature attribution (n_people x n_features).

    Uses gradient x input, summed over the time axis so each feature gets one
    importance value per person.
    """
    X = X.clone().detach().requires_grad_(True)
    model.eval()
    pred = model(X)
    pred.sum().backward()
    grad = X.grad.detach().numpy()          # (people, frames, features)
    inp = X.detach().numpy()
    attr = (grad * inp).sum(axis=1)          # sum over time -> (people, features)
    return attr


def main():
    sequences, info = build_sequences(personalise=True)
    X_train, _, _ = make_tensors(sequences, info, "train")
    X_test, y_test, test_rows = make_tensors(sequences, info, "test")
    (X_train, X_test), _ = standardise(X_train, X_test)

    model = GRURegressor(n_features=len(FEATURES))
    model.load_state_dict(torch.load(RESULTS_DIR / f"model_{MODE}.pt"))

    attr = attributions(model, X_test)

    # --- Global importance: mean absolute attribution per feature ---
    global_imp = np.abs(attr).mean(axis=0)
    order = np.argsort(global_imp)[::-1]
    print("=== Global feature importance (mean |attribution|) ===")
    for i in order:
        print(f"  {FEATURES[i]:<20} {global_imp[i]:.3f}")

    # Save a bar chart.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([FEATURES[i] for i in order][::-1],
            [global_imp[i] for i in order][::-1])
    ax.set_xlabel("Mean |attribution|")
    ax.set_title("Global feature importance (personalised model)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=120)
    print("\nSaved chart to results/feature_importance.png")

    # --- Per-person explanation for one example ---
    example = 0
    person_attr = attr[example]
    pid = int(test_rows.loc[example, "participant_id"])
    true = float(y_test[example])
    # Most influential feature for this person.
    top = np.argsort(np.abs(person_attr))[::-1][:2]
    print(f"\n=== Example explanation (participant {pid}, true PHQ-8 ={true:.0f}) ===")
    for i in top:
        direction = "raising" if person_attr[i] > 0 else "lowering"
        print(f"  {READABLE[FEATURES[i]]}: {direction} the predicted score "
              f"(attribution {person_attr[i]:+.2f})")


if __name__ == "__main__":
    main()