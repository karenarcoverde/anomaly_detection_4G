"""Train the final LightGBM model and save it to the BentoML model store."""

import argparse
from pathlib import Path
from typing import Tuple

import bentoml
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold


TRAIN_FILENAME = "ML-MATT-CompetitionQT2021_train.csv"
TARGET = "Unusual"
METRIC_COLUMNS = [
    "PRBUsageUL",
    "PRBUsageDL",
    "meanThr_DL",
    "meanThr_UL",
    "maxThr_DL",
    "maxThr_UL",
    "meanUE_DL",
    "meanUE_UL",
    "maxUE_DL",
    "maxUE_UL",
]
FEATURE_COLUMNS = METRIC_COLUMNS + ["Hour"]
N_SPLITS = 3

# These are the best parameters reported by the LightGBM search in notebook.ipynb.
MODEL_PARAMS = {
    "objective": "binary",
    "random_state": 42,
    "n_jobs": 4,
    "verbosity": -1,
    "scale_pos_weight": 2,
    "reg_lambda": 3,
    "num_leaves": 15,
    "n_estimators": 900,
    "min_child_samples": 20,
    "max_depth": 6,
    "learning_rate": 0.1,
}


def load_training_data(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load, clean, and prepare training features using the notebook's logic."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            "Training CSV not found: {}. Download it from Kaggle and place it "
            "in the project folder, or pass its path with --data.".format(csv_path)
        )

    data = pd.read_csv(csv_path, sep=";")
    required_columns = (
        set(METRIC_COLUMNS)
        | {"Time", "CellName", "maxUE_UL+DL", TARGET}
    )
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Training CSV is missing required columns: {}".format(
                ", ".join(sorted(missing_columns))
            )
        )

    data = data.drop_duplicates().reset_index(drop=True)
    time = pd.to_datetime(data["Time"], format="%H:%M", errors="raise")
    data["Hour"] = time.dt.hour + time.dt.minute / 60

    features = data[FEATURE_COLUMNS]
    target = data[TARGET]
    if target.isnull().any() or set(target.unique()) != {0, 1}:
        raise ValueError("The Unusual target must contain both labels 0 and 1.")

    return features, target.astype(int)


def choose_threshold(features: pd.DataFrame, target: pd.Series) -> float:
    """Choose an F1-maximizing threshold from out-of-fold predictions."""
    class_counts = target.value_counts()
    if class_counts.min() < N_SPLITS:
        raise ValueError(
            "Each target class needs at least {} rows for {}-fold validation.".format(
                N_SPLITS, N_SPLITS
            )
        )

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof_probabilities = np.zeros(len(target), dtype=float)

    for train_indices, validation_indices in cv.split(features, target):
        fold_model = LGBMClassifier(**MODEL_PARAMS)
        fold_model.fit(
            features.iloc[train_indices],
            target.iloc[train_indices],
        )
        oof_probabilities[validation_indices] = fold_model.predict_proba(
            features.iloc[validation_indices]
        )[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        target, oof_probabilities
    )
    precision, recall = precision[:-1], recall[:-1]
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[int(np.argmax(f1_scores))])


def train_and_save(data_path: Path, model_name: str) -> str:
    features, target = load_training_data(data_path)
    threshold = choose_threshold(features, target)

    model = LGBMClassifier(**MODEL_PARAMS)
    model.fit(features, target)

    saved_model = bentoml.lightgbm.save_model(
        model_name,
        model,
        custom_objects={
            "threshold": threshold,
            "feature_names": list(features.columns),
        },
        metadata={
            "target": TARGET,
            "threshold": str(threshold),
            "training_rows": str(len(features)),
        },
    )
    return str(saved_model.tag)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train the final LightGBM model and save it with BentoML."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=project_dir / TRAIN_FILENAME,
        help="Path to the Kaggle training CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--model-name",
        default="anomaly_detection_4g_lgbm",
        help="Name used for the BentoML model (default: %(default)s).",
    )
    args = parser.parse_args()

    model_tag = train_and_save(args.data, args.model_name)
    print("BentoML model saved as: {}".format(model_tag))


if __name__ == "__main__":
    main()
