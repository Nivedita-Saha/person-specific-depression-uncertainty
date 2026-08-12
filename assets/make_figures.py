"""
Generate the headline figures for the README.

Produces three committed PNGs in assets/:
  1. model_comparison.png       - three-way MAE comparison
  2. predictions_intervals.png  - per-person predictions with conformal intervals
  3. feature_importance.png     - which features drive predictions

Run from the project root:
    python -m assets.make_figures
"""

from pathlib import Path
import numpy as np
import contextlib
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import baseline as baseline_mod
from src import model as model_mod
from src.personalise import build_sequences, FEATURES
from src.model import GRURegressor, make_tensors, standardise
from src.explain import attributions
import torch

ASSETS = Path("assets")
RESULTS = Path("results")
BLUE, GREY, GREEN = "#3B6EA5", "#B0B0B0", "#2E8B6F"


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def fig_model_comparison():
    parts = baseline_mod.load_split()
    X_tr, y_tr, _ = parts["train"]
    X_te, y_te, _ = parts["test"]
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    base = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_tr, y_tr)
    base_mae = baseline_mod.evaluate(base, X_te, y_te)[0]
    raw_mae = quiet(model_mod.train, "raw")
    pers_mae = quiet(model_mod.train, "personalised")

    names = ["Averaging\nbaseline", "Temporal\n(raw)", "Temporal\n(personalised)"]
    vals = [base_mae, raw_mae, pers_mae]
    colors = [GREY, GREY, GREEN]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.2f}",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Test MAE (PHQ-8 points) — lower is better")
    ax.set_title("Person-specific modelling beats the baseline;\nraw temporal modelling does not")
    ax.set_ylim(0, max(vals) + 0.8)
    fig.tight_layout()
    fig.savefig(ASSETS / "model_comparison.png", dpi=130)
    plt.close(fig)
    print("Saved assets/model_comparison.png")


def fig_predictions_intervals():
    unc = np.load(RESULTS / "uncertainty_personalised.npz")
    y = unc["y_true"]; mean = unc["pred_mean"]
    lower = unc["lower"]; upper = unc["upper"]; abstain = unc["abstain"]

    order = np.argsort(y)
    y, mean = y[order], mean[order]
    lower, upper, abstain = lower[order], upper[order], abstain[order]
    x = np.arange(len(y))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.fill_between(x, lower, upper, color=BLUE, alpha=0.15,
                    label="90% conformal interval")
    ax.plot(x, mean, "o", color=BLUE, markersize=5, label="Predicted PHQ-8")
    ax.plot(x, y, "x", color="black", markersize=6, label="True PHQ-8")
    if abstain.any():
        ax.scatter(x[abstain], mean[abstain], s=140, facecolors="none",
                   edgecolors="crimson", linewidths=1.6,
                   label="Flagged for clinician review")
    ax.set_xlabel("Test participants (sorted by true PHQ-8)")
    ax.set_ylabel("PHQ-8 score")
    ax.set_title("Predictions with calibrated uncertainty intervals")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSETS / "predictions_intervals.png", dpi=130)
    plt.close(fig)
    print("Saved assets/predictions_intervals.png")


def fig_feature_importance():
    sequences, info = build_sequences(personalise=True)
    X_train, _, _ = make_tensors(sequences, info, "train")
    X_test, _, _ = make_tensors(sequences, info, "test")
    (X_train, X_test), _ = standardise(X_train, X_test)
    model = GRURegressor(n_features=len(FEATURES))
    model.load_state_dict(torch.load(RESULTS / "model_personalised.pt"))
    attr = np.abs(attributions(model, X_test)).mean(axis=0)
    order = np.argsort(attr)

    # Highlight the features that actually carry planted signal.
    signal = {"AU06_cheek_raiser", "AU12_lip_corner", "AU04_brow_lowerer",
              "gaze_y", "pose_Rx"}
    colors = [GREEN if FEATURES[i] in signal else GREY for i in order]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh([FEATURES[i] for i in order], [attr[i] for i in order], color=colors)
    ax.set_xlabel("Mean |attribution|")
    ax.set_title("Feature importance (green = features carrying true signal)")
    fig.tight_layout()
    fig.savefig(ASSETS / "feature_importance.png", dpi=130)
    plt.close(fig)
    print("Saved assets/feature_importance.png")


def main():
    ASSETS.mkdir(exist_ok=True)
    fig_model_comparison()
    fig_predictions_intervals()
    fig_feature_importance()
    print("\nAll figures written to assets/")


if __name__ == "__main__":
    main()