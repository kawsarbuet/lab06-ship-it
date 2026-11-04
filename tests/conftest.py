"""Shared test data for the Lab 05 model's five features."""

import pytest


@pytest.fixture
def valid_order():
    return {
        "distance_km": 5,
        "items": 3,
        "hour": 16,
        "weather": "clear",
        "day_type": "weekday",
    }
