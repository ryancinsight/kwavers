import importlib.util
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "python" / "pykwavers" / "parity_targets.py"
    spec = importlib.util.spec_from_file_location("parity_targets", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_quick_tier_detects_failure():
    module = _load_module()
    result = module.evaluate_parity(
        {"pearson_r": 0.94, "rms_ratio": 0.75, "psnr_db": 25.6},
        "fundamental",
        16,
    )

    assert result["tier"] == "quick"
    assert result["status"] == "FAIL"
    assert result["checks"]["pearson_r"] is False


def test_full_tier_detects_pass():
    module = _load_module()
    result = module.evaluate_parity(
        {"pearson_r": 0.99, "rms_ratio": 1.0, "psnr_db": 35.0},
        "fundamental",
        module.N_SCAN_LINES_DEFAULT,
    )

    assert result["tier"] == "full"
    assert result["status"] == "PASS"
    assert all(result["checks"].values())
