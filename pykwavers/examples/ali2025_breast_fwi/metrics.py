"""PyO3 wrappers for Rust-owned Ali 2025 reconstruction metrics."""

from __future__ import annotations

import numpy as np


ALI_2025_TABLE1_3D_FWI = {
    1: {"rmse_m_s": 15.5, "pearson_correlation": 0.8848},
    2: {"rmse_m_s": 10.1, "pearson_correlation": 0.8981},
    3: {"rmse_m_s": 8.4, "pearson_correlation": 0.8967},
}


def rmse_m_s(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(reconstruction_metrics(reference, estimate)["rmse_m_s"])


def normalized_rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(reconstruction_metrics(reference, estimate)["normalized_rmse"])


def pearson_correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(reconstruction_metrics(reference, estimate)["pearson_correlation"])


def reconstruction_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    return dict(
        _kw().breast_fwi_reconstruction_metrics(
            np.asarray(reference, dtype=np.float64),
            np.asarray(estimate, dtype=np.float64),
        )
    )


def table1_parity(
    metrics: dict[str, float],
    phantom_index: int,
    rmse_multiplier: float,
    pcc_fraction: float,
) -> dict[str, float | bool]:
    return dict(
        _kw().breast_fwi_table1_parity(
            float(metrics["rmse_m_s"]),
            float(metrics["pearson_correlation"]),
            int(phantom_index),
            float(rmse_multiplier),
            float(pcc_fraction),
        )
    )


def _kw():
    import pykwavers as kw

    return kw
