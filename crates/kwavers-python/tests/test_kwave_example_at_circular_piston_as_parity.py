"""Parity regression for the vendored `at_circular_piston_AS` example."""

from __future__ import annotations

import os
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import requires_kwave
from parity_test_utils import assert_decodable_nonblank_png


skip_kwave = os.getenv("KWAVERS_SKIP_KWAVE", "0") == "1"
run_slow = os.getenv("KWAVERS_RUN_SLOW", "0") == "1"
slow_reason = "Set KWAVERS_RUN_SLOW=1 to run slow axisymmetric k-wave-python tests"


def _metric_value(text: str, label: str) -> float:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(label):
            if "=" in stripped:
                payload = stripped.split("=", 1)[1]
            elif ":" in stripped:
                payload = stripped.split(":", 1)[1]
            else:
                raise AssertionError(f"missing separator in metric line: {label}")
            return float(payload.split()[0])
    raise AssertionError(f"missing metric line: {label}")


def _run_example() -> Path:
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "at_circular_piston_AS_compare.py"
    metrics_path = root / "examples" / "output" / "at_circular_piston_AS_metrics.txt"
    env = os.environ.copy()
    env["KWAVERS_RUN_SLOW"] = "1"
    env["PYKWAVERS_EXTENSION_PATH"] = str(root / "target" / "maturin" / "pykwavers.dll")
    subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return metrics_path


def _load_example_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "examples" / "at_circular_piston_AS_compare.py"
    examples_dir = str(root / "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location(
        "at_circular_piston_AS_compare",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_report_contract(module, text: str):
    thresholds = module.PARITY_THRESHOLDS
    pearson_r = _metric_value(text, "pearson_r")
    rms_ratio = _metric_value(text, "rms_ratio")
    psnr_db = _metric_value(text, "psnr_db")
    kwave_vs_analytical = _metric_value(text, "kwave_vs_analytical_pearson")
    pkwav_vs_analytical = _metric_value(text, "pkwav_vs_analytical_pearson")
    analytical_thresholds = module.ANALYTICAL_REFERENCE_THRESHOLDS

    assert "parity_status: PASS" in text
    assert pearson_r >= thresholds["pearson_r"]
    assert thresholds["rms_ratio_min"] <= rms_ratio
    assert rms_ratio <= thresholds["rms_ratio_max"]
    assert psnr_db >= thresholds["psnr_db"]
    assert kwave_vs_analytical > analytical_thresholds["kwave_pearson_min"]
    assert pkwav_vs_analytical > analytical_thresholds["pykwavers_pearson_min"]
    assert (
        abs(pkwav_vs_analytical - kwave_vs_analytical)
        < analytical_thresholds["pearson_agreement_abs_max"]
    )


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
def test_current_at_circular_piston_as_artifacts_match_thresholds():
    module = _load_example_module()
    text = module.METRICS_PATH.read_text(encoding="utf-8")

    assert module.METRICS_PATH.exists()
    assert module.FIGURE_PATH.exists()
    _assert_report_contract(module, text)
    assert_decodable_nonblank_png(module.FIGURE_PATH)


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.skipif(not run_slow, reason=slow_reason)
@pytest.mark.timeout(300)
class TestKWaveExampleParityAtCircularPistonAS:
    def test_at_circular_piston_as_metrics(self):
        module = _load_example_module()
        metrics_path = _run_example()
        figure_path = module.FIGURE_PATH
        text = metrics_path.read_text(encoding="utf-8")

        kw_runtime = _metric_value(text, "kwave_runtime_s")
        pkw_runtime = _metric_value(text, "pykwavers_runtime_s")

        assert metrics_path.exists()
        assert figure_path.exists()
        _assert_report_contract(module, text)
        assert kw_runtime > 0.0
        assert pkw_runtime > 0.0
        assert pkw_runtime < kw_runtime
        assert_decodable_nonblank_png(figure_path)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
