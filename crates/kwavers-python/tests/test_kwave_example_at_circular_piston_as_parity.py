"""Parity regression for the vendored `at_circular_piston_AS` example."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import requires_kwave


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


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.skipif(not run_slow, reason=slow_reason)
@pytest.mark.timeout(300)
class TestKWaveExampleParityAtCircularPistonAS:
    def test_at_circular_piston_as_metrics(self):
        metrics_path = _run_example()
        figure_path = metrics_path.with_name("at_circular_piston_AS_compare.png")
        text = metrics_path.read_text(encoding="utf-8")

        pearson_r = _metric_value(text, "pearson_r")
        rms_ratio = _metric_value(text, "rms_ratio")
        psnr_db = _metric_value(text, "psnr_db")
        kwave_vs_analytical = _metric_value(text, "kwave_vs_analytical_pearson")
        pkwav_vs_analytical = _metric_value(text, "pkwav_vs_analytical_pearson")
        kw_runtime = _metric_value(text, "kwave_runtime_s")
        pkw_runtime = _metric_value(text, "pykwavers_runtime_s")

        assert metrics_path.exists()
        assert figure_path.exists()
        assert "parity_status: PASS" in text
        assert pearson_r > 0.99
        assert abs(rms_ratio - 1.0) < 5e-2
        assert psnr_db > 20.0
        assert kwave_vs_analytical > 0.99
        assert pkwav_vs_analytical > 0.98
        assert pkwav_vs_analytical <= kwave_vs_analytical
        assert kw_runtime > 0.0
        assert pkw_runtime > 0.0
        assert pkw_runtime < kw_runtime


if __name__ == "__main__":
    pytest.main(["-v", __file__])
