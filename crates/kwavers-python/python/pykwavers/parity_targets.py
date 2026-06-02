"""Shared scientific parity targets for pykwavers validation workflows."""

N_SCAN_LINES_DEFAULT = 96

PARITY_THRESHOLDS = {
    "quick": {
        "fundamental": {"pearson_r": 0.97, "rms_ratio_min": 0.90, "rms_ratio_max": 1.10, "psnr_db": 28.0},
        "harmonic": {"pearson_r": 0.92, "rms_ratio_min": 0.85, "rms_ratio_max": 1.15, "psnr_db": 24.0},
    },
    "full": {
        "fundamental": {"pearson_r": 0.985, "rms_ratio_min": 0.95, "rms_ratio_max": 1.05, "psnr_db": 32.0},
        "harmonic": {"pearson_r": 0.95, "rms_ratio_min": 0.90, "rms_ratio_max": 1.10, "psnr_db": 26.0},
    },
}


def evaluate_parity(metrics, label, n_lines, full_scan_lines=N_SCAN_LINES_DEFAULT):
    """Evaluate measured parity against tiered scientific targets."""
    tier = "quick" if n_lines < full_scan_lines else "full"
    target = PARITY_THRESHOLDS[tier][label]
    checks = {
        "pearson_r": metrics["pearson_r"] >= target["pearson_r"],
        "rms_ratio": target["rms_ratio_min"] <= metrics["rms_ratio"] <= target["rms_ratio_max"],
        "psnr_db": metrics["psnr_db"] >= target["psnr_db"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {"tier": tier, "target": target, "checks": checks, "status": status}
