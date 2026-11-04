"""Protect QuickBite input validation and the cost-based action."""

import importlib.util

import joblib
import pytest

import predict


def test_probabilities_and_ordering():
    easy = {
        "distance_km": 1,
        "items": 1,
        "hour": 11,
        "weather": "clear",
        "day_type": "weekday",
    }
    hard = {
        "distance_km": 25,
        "items": 19,
        "hour": 21,
        "weather": "snow",
        "day_type": "weekend",
    }
    p_easy = predict.predict_order(easy)["late_probability"]
    p_hard = predict.predict_order(hard)["late_probability"]
    assert 0 <= p_easy < p_hard <= 1


def test_voucher_boundary():
    t = predict.THRESHOLD
    assert predict.should_offer_voucher(t)
    assert not predict.should_offer_voucher(t - 0.00001)
    assert predict.should_offer_voucher(t + 0.00001)


def test_threshold_comes_from_metadata(valid_order):
    result = predict.predict_order(valid_order)
    assert result["threshold"] == predict.MODEL_INFO["threshold"]
    assert result["offer_voucher"] == (
        result["late_probability"] >= result["threshold"]
    )


def test_optional_prep_if_present_in_schema(valid_order):
    if "prep_minutes" not in predict.FEATURES:
        pytest.skip("Lab 05 model has no prep_minutes feature")
    valid_order["prep_minutes"] = None
    result = predict.predict_order(valid_order)
    assert result["warnings"]


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("distance_km", 0),
        ("distance_km", 31),
        ("items", 0),
        ("items", 2.3),
        ("hour", 9),
        ("hour", 23),
        ("distance_km", float("nan")),
    ],
)
def test_invalid_numbers_rejected(valid_order, field, bad):
    valid_order[field] = bad
    with pytest.raises(ValueError, match=field):
        predict.predict_order(valid_order)


def test_missing_field_rejected(valid_order):
    del valid_order["weather"]
    with pytest.raises(ValueError, match="weather"):
        predict.predict_order(valid_order)


def test_unseen_category_rejected(valid_order):
    valid_order["weather"] = "Snow"
    with pytest.raises(ValueError, match="weather"):
        predict.predict_order(valid_order)


def test_unexpected_field_rejected(valid_order):
    valid_order["vehicle"] = "bike"
    with pytest.raises(ValueError, match="vehicle"):
        predict.predict_order(valid_order)


def test_model_loaded_once(monkeypatch):
    count = 0
    original = joblib.load

    def counting_load(*args, **kwargs):
        nonlocal count
        count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(joblib, "load", counting_load)
    spec = importlib.util.spec_from_file_location(
        "predict_once_probe", predict.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert count == 1
    module.predict_order(
        {
            "distance_km": 3,
            "items": 2,
            "hour": 14,
            "weather": "clear",
            "day_type": "weekday",
        }
    )
    assert count == 1


def test_case_sensitive_weather_validation(valid_order):
    valid_order["weather"] = "snow"
    assert 0 <= predict.predict_order(valid_order)["late_probability"] <= 1
    valid_order["weather"] = "Snow"
    with pytest.raises(ValueError, match="weather"):
        predict.predict_order(valid_order)
