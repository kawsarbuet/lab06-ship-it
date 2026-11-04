"""Validated inference for QuickBite, using the exact Lab 05 feature schema."""

import json
import math
from pathlib import Path

import joblib
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_INFO = json.loads((MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))
if sklearn.__version__ != MODEL_INFO["scikit_learn_version"]:
    raise RuntimeError(
        "scikit-learn version mismatch: model needs "
        f"{MODEL_INFO['scikit_learn_version']}, found {sklearn.__version__}"
    )
PIPELINE = joblib.load(MODEL_DIR / "late_orders.joblib")
THRESHOLD = float(MODEL_INFO["threshold"])

# Lab 06 bounds are only applied to features PRESENT in the trained Lab 05 schema.
NUMERIC_RULES = {
    "distance_km": (0.1, 30.0, False),
    "items": (1, 20, True),
    "hour": (10, 22, True),
    "prep_minutes": (1, 60, False),
    "courier_months": (0, 120, False),
}
CATEGORY_RULES = {
    "weather": ("clear", "rain", "snow"),
    "day_type": ("weekday", "weekend"),
    "vehicle": ("bike", "scooter", "car"),
}
FEATURES = tuple(MODEL_INFO["features"])
NUMERIC_FEATURES = tuple(MODEL_INFO["numeric_features"])
CATEGORICAL_FEATURES = tuple(MODEL_INFO["categorical_features"])
if set(FEATURES) != set(NUMERIC_FEATURES + CATEGORICAL_FEATURES):
    raise RuntimeError("Feature lists in model_info.json disagree")
if any(f not in NUMERIC_RULES for f in NUMERIC_FEATURES) or any(
    f not in CATEGORY_RULES for f in CATEGORICAL_FEATURES
):
    raise RuntimeError("Unrecognized model feature: add an explicit validation rule")


def validate_order(order: dict) -> tuple[dict, list[str]]:
    """Return checked values and warnings; reject missing/invalid fields."""
    if not isinstance(order, dict):
        raise ValueError("order must be a dictionary of field values")
    unexpected = set(order) - set(FEATURES)
    if unexpected:
        raise ValueError(f"Unexpected field(s): {', '.join(sorted(unexpected))}")
    clean = {}
    warnings = []
    for field in FEATURES:
        value = order.get(field)
        missing = value is None or (isinstance(value, str) and not value.strip())
        if missing:
            if field == "prep_minutes":
                clean[field] = float("nan")
                warnings.append(
                    "prep_minutes was empty; the saved pipeline imputes its median."
                )
                continue
            raise ValueError(f"{field}: a value is required")
        if field in NUMERIC_FEATURES:
            low, high, whole = NUMERIC_RULES[field]
            if isinstance(value, bool):
                raise ValueError(f"{field}: enter a number from {low:g} to {high:g}")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{field}: enter a number from {low:g} to {high:g}"
                ) from exc
            if not math.isfinite(number) or not low <= number <= high:
                raise ValueError(f"{field}: must be from {low:g} to {high:g}")
            if whole and not number.is_integer():
                raise ValueError(f"{field}: must be a whole number")
            clean[field] = int(number) if whole else number
        else:
            allowed = CATEGORY_RULES[field]
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"{field}: must be one of {', '.join(allowed)}")
            clean[field] = value
    return clean, warnings


def should_offer_voucher(probability: float, threshold: float | None = None) -> bool:
    """Boundary is inclusive: offer the voucher at probability >= threshold."""
    if threshold is None:
        threshold = THRESHOLD
    if not 0 <= probability <= 1 or not 0 <= threshold <= 1:
        raise ValueError("probability and threshold must be between 0 and 1")
    return probability >= threshold


def predict_order(order: dict) -> dict:
    """Validate, score and return outcome probabilities and business decision."""
    checked, warnings = validate_order(order)
    probabilities = PIPELINE.predict_proba(pd.DataFrame([checked], columns=FEATURES))[0]
    classes = list(PIPELINE.classes_)
    late_index = classes.index(1)
    late_probability = float(probabilities[late_index])
    return {
        "late_probability": late_probability,
        "on_time_probability": 1.0 - late_probability,
        "offer_voucher": should_offer_voucher(late_probability),
        "threshold": THRESHOLD,
        "warnings": warnings,
    }
