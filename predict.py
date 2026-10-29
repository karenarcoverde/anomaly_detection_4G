"""Serve the trained anomaly detector as a BentoML CSV upload API."""

import os
from functools import lru_cache
from io import BytesIO
from typing import Any, BinaryIO, Dict, List

import bentoml
import numpy as np
import pandas as pd
from bentoml.io import File


DEFAULT_MODEL_TAG = "anomaly_detection_4g_lgbm:latest"
MODEL_TAG = os.environ.get("BENTOML_MODEL_TAG", DEFAULT_MODEL_TAG)
PREDICTION_COLUMN = "Unusual"
HOUR_COLUMN = "Hour"


def prepare_features(
    records: List[Dict[str, Any]], feature_names: List[str]
) -> pd.DataFrame:
    """Validate raw request rows and prepare features in training order."""
    if not records:
        raise ValueError("The request must contain at least one record.")

    data = pd.DataFrame.from_records(records)
    if "Time" in data.columns:
        time = pd.to_datetime(data["Time"], format="%H:%M", errors="raise")
        data[HOUR_COLUMN] = time.dt.hour + time.dt.minute / 60
    elif HOUR_COLUMN not in data.columns:
        raise ValueError("Each record must include either Time or Hour.")

    required_columns = [name for name in feature_names if name != HOUR_COLUMN]
    missing_columns = set(required_columns).difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Request records are missing required columns: {}".format(
                ", ".join(sorted(missing_columns))
            )
        )
    if HOUR_COLUMN in feature_names and HOUR_COLUMN not in data.columns:
        raise ValueError("Each record must include either Time or Hour.")

    features = data.loc[:, feature_names].apply(pd.to_numeric, errors="raise")
    if features.isnull().any().any():
        raise ValueError("Feature values must not be null.")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Feature values must be finite numbers.")

    return features


def load_model(model_tag: str) -> tuple[Any, float, List[str]]:
    """Load the model and its inference metadata from BentoML."""
    model_info = bentoml.models.get(model_tag)
    model = bentoml.lightgbm.load_model(model_info)
    threshold = float(model_info.info.metadata["threshold"])
    feature_names = model_info.custom_objects["feature_names"]
    return model, threshold, feature_names


@lru_cache(maxsize=1)
def load_service_model() -> tuple[Any, float, List[str]]:
    """Load the selected model once when the first service request arrives."""
    return load_model(MODEL_TAG)


def add_predictions(
    test_data: pd.DataFrame,
    model: Any,
    threshold: float,
    feature_names: List[str],
) -> pd.DataFrame:
    """Return a copy of the input rows with model predictions appended."""
    if test_data.empty:
        raise ValueError("The test CSV contains no rows.")

    features = prepare_features(
        test_data.to_dict(orient="records"), feature_names
    )
    probabilities = model.predict(features)
    predicted_data = test_data.copy()
    predicted_data[PREDICTION_COLUMN] = (
        probabilities >= threshold
    ).astype(int)
    return predicted_data


service = bentoml.legacy.Service("anomaly-detection-4g", runners=[])


@service.api(
    input=File(),
    output=File(mime_type="text/csv"),
    name="predict",
    route="/predict",
)
def predict(input_file: BinaryIO) -> BytesIO:
    """Return the uploaded CSV with model predictions in the Unusual column."""
    test_data = pd.read_csv(BytesIO(input_file.read()), sep=";")
    model, threshold, feature_names = load_service_model()
    predicted_data = add_predictions(
        test_data, model, threshold, feature_names
    )
    return BytesIO(predicted_data.to_csv(sep=";", index=False).encode("utf-8"))
