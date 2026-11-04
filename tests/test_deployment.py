"""Keep Hugging Face settings, requirements and training environment aligned."""

import importlib.metadata
import re
from pathlib import Path

import gradio
import sklearn

from predict import MODEL_INFO

ROOT = Path(__file__).resolve().parents[1]


def test_sklearn_runtime_and_pin():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    expected = MODEL_INFO["scikit_learn_version"]
    assert sklearn.__version__ == expected
    assert f"scikit-learn=={expected}" in req.splitlines()


def test_gradio_not_in_space_requirements():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert not any(line.startswith("gradio") for line in req.splitlines())


def test_readme_sdk_version_and_app_file():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    sdk_version = re.search(r"^sdk_version:\s*(\S+)", readme, re.M).group(1)
    app_file = re.search(r"^app_file:\s*(\S+)", readme, re.M).group(1)
    assert sdk_version == gradio.__version__
    assert (ROOT / app_file).is_file()


def test_requirements_dev_pins():
    dev = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert dev.splitlines()[0] == "-r requirements.txt"
    for package in ("gradio", "pytest", "ruff"):
        assert re.search(rf"^{package}==[\d.]+$", dev, re.M)


def test_model_files_exist():
    assert (ROOT / "models" / "late_orders.joblib").is_file()
    assert (ROOT / "models" / "model_info.json").is_file()


def test_gradio_version_is_pinned_in_dev_requirements():
    dev = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert f"gradio=={importlib.metadata.version('gradio')}" in dev
