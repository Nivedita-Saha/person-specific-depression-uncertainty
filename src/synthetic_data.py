"""
Synthetic data generator for the person-specific depression project.

Creates fake participants whose expressive-behaviour features carry a depression
signal, so the whole pipeline can be built and tested before the real
DAIC-WOZ / E-DAIC data is available. The feature names mimic OpenFace-style
outputs (facial action units, gaze, head pose) so swapping in real data later
is straightforward.

Run from the project root:
    python -m src.synthetic_data
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---- configuration ----
SEED = 42
N_PARTICIPANTS = 200
N_FRAMES = 120          # length of each downsampled "session"
N_CALIBRATION = 30      # first frames = the person's neutral baseline segment

FEATURES = [
    "AU06_cheek_raiser",   # positive affect (smiling)
    "AU12_lip_corner",     # positive affect (smiling)
    "AU01_inner_brow",     # surprise / attention
    "AU04_brow_lowerer",   # negative affect
    "gaze_x",
    "gaze_y",
    "pose_Rx",
    "pose_Ry",
]

OUT_DIR = Path("data/synthetic")


def make_participant(rng, severity):
    """Return a (N_FRAMES x n_features) array of behaviour for one person.

    `severity` is in [0, 1] (0 = not depressed, 1 = severe). Higher severity
    means: dampened positive-expression reactions (blunted affect), slightly
    raised negative expression, and mild downward gaze/head shift.
    """
    n_feat = len(FEATURES)

    # Each person has their own idiosyncratic neutral level: this is the
    # personal baseline the personalisation step will later subtract out.
    person_offset = rng.normal(0.0, 0.30, size=n_feat)

    # Start every frame at the person's baseline plus small measurement noise.
    noise_sd = 0.20 * (1.0 - 0.4 * severity)          # blunted affect = less variability
    frames = person_offset + rng.normal(0.0, noise_sd, size=(N_FRAMES, n_feat))

    # Interaction phase = everything after the calibration segment.
    interaction = np.arange(N_FRAMES) >= N_CALIBRATION
    t = np.linspace(0, 6 * np.pi, N_FRAMES)           # a slow oscillation over the session

    # Positive-affect AUs (indices 0 and 1): reactive bumps during interaction,
    # with amplitude reduced as severity rises.
    pos_amp = 1.0 * (1.0 - 0.7 * severity)
    reaction = pos_amp * (0.5 + 0.5 * np.sin(t))
    frames[interaction, 0] += reaction[interaction]
    frames[interaction, 1] += reaction[interaction]

    # Negative-affect AU (index 3): mild sustained increase with severity.
    frames[interaction, 3] += 0.5 * severity

    # Gaze / head pose (indices 5 and 6): mild downward shift with severity.
    frames[interaction, 5] += 0.4 * severity
    frames[interaction, 6] += 0.3 * severity

    return frames


def generate():
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_rows = []
    label_rows = []

    for pid in range(N_PARTICIPANTS):
        # Draw a latent severity skewed toward lower values (most people not depressed).
        severity = rng.beta(2.0, 3.0)
        phq8 = int(round(severity * 24))              # PHQ-8 total score, 0-24
        depressed = int(phq8 >= 10)                   # standard clinical cut-off
        gender = rng.choice(["F", "M"], p=[0.45, 0.55])  # mild imbalance, as in real data

        frames = make_participant(rng, severity)

        for f in range(N_FRAMES):
            row = {
                "participant_id": pid,
                "frame": f,
                "phase": "calibration" if f < N_CALIBRATION else "interaction",
            }
            for j, name in enumerate(FEATURES):
                row[name] = frames[f, j]
            feature_rows.append(row)

        label_rows.append({
            "participant_id": pid,
            "phq8": phq8,
            "depressed": depressed,
            "gender": gender,
        })

    features_df = pd.DataFrame(feature_rows)
    labels_df = pd.DataFrame(label_rows)

    # Reproducible train / val / test split by participant (70 / 15 / 15).
    ids = labels_df["participant_id"].to_numpy()
    shuffled = rng.permutation(ids)
    n = len(shuffled)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    split = {}
    for i, pid in enumerate(shuffled):
        if i < n_train:
            split[pid] = "train"
        elif i < n_train + n_val:
            split[pid] = "val"
        else:
            split[pid] = "test"
    labels_df["split"] = labels_df["participant_id"].map(split)

    features_df.to_csv(OUT_DIR / "features.csv", index=False)
    labels_df.to_csv(OUT_DIR / "labels.csv", index=False)

    # Print a short summary so we can sanity-check what was generated.
    print("Synthetic data written to:", OUT_DIR.resolve())
    print("Participants:", len(labels_df))
    print("Feature rows:", len(features_df))
    print("PHQ-8 range:", labels_df["phq8"].min(), "-", labels_df["phq8"].max())
    print("Depressed (PHQ-8 >= 10):", int(labels_df["depressed"].sum()),
          "of", len(labels_df))
    print("Gender counts:\n", labels_df["gender"].value_counts().to_string())
    print("Split counts:\n", labels_df["split"].value_counts().to_string())


if __name__ == "__main__":
    generate()