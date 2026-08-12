"""
Honest baseline: a direct behaviour-to-label model.

For each participant, average every feature across the whole session into a
single feature vector, then predict PHQ-8 severity with a standard regressor.
No temporal information and no personalisation - this is the plain baseline
that the person-specific temporal model must beat to justify its complexity.

Run from the project root:
    python -m src.baseline
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = Path("data/synthetic")
FEATURES = [
    "AU06_cheek_raiser", "AU12_lip_corner", "AU01_inner_brow",
    "AU04_brow_lowerer", "gaze_x", "gaze_y", "pose_Rx", "pose_Ry",
]


def build_session_averages(features_df):
    """Collapse each participant's session to one row of averaged features."""
    return (
        features_df
        .groupby("participant_id")[FEATURES]
        .mean()
        .reset_index()
    )


def load_split():
    features = pd.read_csv(DATA_DIR / "features.csv")
    labels = pd.read_csv(DATA_DIR / "labels.csv")

    averaged = build_session_averages(features)
    data = averaged.merge(labels, on="participant_id")

    parts = {}
    for name in ["train", "val", "test"]:
        subset = data[data["split"] == name]
        X = subset[FEATURES].to_numpy()
        y = subset["phq8"].to_numpy()
        parts[name] = (X, y, subset)
    return parts


def evaluate(model, X, y):
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    # Secondary: turn predicted score into a depression label at cut-off 10.
    pred_label = (pred >= 10).astype(int)
    true_label = (y >= 10).astype(int)
    acc = (pred_label == true_label).mean()
    return mae, rmse, acc


def main():
    parts = load_split()
    X_train, y_train, _ = parts["train"]
    X_val, y_val, _ = parts["val"]
    X_test, y_test, _ = parts["test"]

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X_train, y_train)

    print("=== Baseline: session-averaged features -> PHQ-8 (Ridge) ===")
    for name, (X, y) in [("val", (X_val, y_val)), ("test", (X_test, y_test))]:
        mae, rmse, acc = evaluate(model, X, y)
        print(f"{name:>5}:  MAE={mae:5.2f}   RMSE={rmse:5.2f}   "
              f"depression-accuracy={acc:.2%}")

    # Reference point: how bad is 'always predict the training mean'?
    naive = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
    naive_mae = mean_absolute_error(y_test, naive)
    print(f"\nReference (predict train mean): test MAE={naive_mae:5.2f}")
    print("The model should beat this reference to be doing anything useful.")


if __name__ == "__main__":
    main()