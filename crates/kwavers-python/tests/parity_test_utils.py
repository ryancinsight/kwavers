"""Shared helpers for cached parity artifact tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def load_example_module(script: str) -> ModuleType:
    """Load an example script as a module for direct contract inspection."""
    examples_dir = str(EXAMPLES)
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    module_name = script.removesuffix(".py")
    module_path = EXAMPLES / script
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_numeric_cache(path: Path) -> dict[str, np.ndarray]:
    """Load numeric arrays from a NumPy cache without object deserialization."""
    with np.load(path, allow_pickle=False) as cache:
        return {
            name: np.asarray(cache[name])
            for name in cache.files
            if np.issubdtype(np.asarray(cache[name]).dtype, np.number)
        }


def has_nonzero_payload(arrays: dict[str, np.ndarray]) -> bool:
    """Return true when any non-runtime numeric payload carries signal."""
    for name, array in arrays.items():
        if name == "runtime_s":
            continue
        if array.size <= 1:
            continue
        if np.all(np.isfinite(array)) and float(np.max(np.abs(array))) > 0.0:
            return True
    return False


def assert_decodable_nonblank_png(path: Path):
    """Assert that a generated PNG is decodable, finite, and nonblank."""
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        extrema = image.getextrema()

    assert height >= 32, path.name
    assert width >= 32, path.name
    if extrema and isinstance(extrema[0], tuple):
        assert any(channel_max > channel_min for channel_min, channel_max in extrema), path.name
    else:
        channel_min, channel_max = extrema
        assert channel_max > channel_min, path.name


def report_metric_value(text: str, label: str, section: str | None = None) -> float:
    """Read a numeric metric from an optional report section."""
    lines = text.splitlines()
    start = 0
    if section is not None:
        for index, line in enumerate(lines):
            if line.strip() == section:
                start = index + 1
                break
        else:
            raise AssertionError(f"missing report section: {section}")

    for line in lines[start:]:
        stripped = line.strip()
        if section is not None and stripped.endswith(":") and not stripped.startswith(label):
            raise AssertionError(f"missing metric {label} in section: {section}")
        if stripped.startswith(label):
            if "=" in stripped:
                payload = stripped.split("=", 1)[1]
            elif ":" in stripped:
                payload = stripped.split(":", 1)[1]
            else:
                raise AssertionError(f"missing separator in metric line: {label}")
            return float(payload.split()[0])
    raise AssertionError(f"missing metric line: {label}")


def assert_cached_example_artifacts(module: ModuleType):
    """Assert that a cached parity example has a PASS report and PNG artifact."""
    metrics_path = getattr(module, "METRICS_PATH")
    figure_path = getattr(module, "FIGURE_PATH", getattr(module, "PNG_PATH", None))

    assert metrics_path.exists()
    assert figure_path is not None
    assert figure_path.exists()

    metrics_text = metrics_path.read_text(encoding="utf-8", errors="ignore")
    has_pass_status = (
        "parity_status: PASS" in metrics_text
        or "OVERALL: PASS" in metrics_text
    )
    assert has_pass_status, metrics_path.name
    assert_decodable_nonblank_png(figure_path)
