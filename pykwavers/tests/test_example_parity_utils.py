import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "examples" / "example_parity_utils.py"
    spec = importlib.util.spec_from_file_location("example_parity_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_trace_metrics_identity():
    module = _load_module()
    x = np.linspace(-1.0, 1.0, 128)
    metrics = module.compute_trace_metrics(x, x.copy())
    assert abs(metrics["pearson_r"] - 1.0) < 1e-12
    assert abs(metrics["rms_ratio"] - 1.0) < 1e-12
    assert metrics["rmse"] == 0.0
    assert metrics["max_abs_diff"] == 0.0


def test_image_metrics_identity():
    module = _load_module()
    img = np.arange(64, dtype=float).reshape(8, 8)
    metrics = module.compute_image_metrics(img, img.copy())
    assert abs(metrics["pearson_r"] - 1.0) < 1e-12
    assert abs(metrics["rms_ratio"] - 1.0) < 1e-12
    assert metrics["psnr_db"] > 300.0


def test_sensor_matrix_summary_identity():
    module = _load_module()
    mat = np.arange(24, dtype=float).reshape(4, 6)
    summary = module.summarize_sensor_matrix_metrics(mat, mat.copy(), expected_sensors=4)
    assert summary["n_sensors"] == 4.0
    assert summary["n_time_samples"] == 6.0
    assert abs(summary["pearson_r_mean"] - 1.0) < 1e-12
    assert abs(summary["pearson_r_median"] - 1.0) < 1e-12
    assert abs(summary["rms_ratio_mean"] - 1.0) < 1e-12
    assert abs(summary["rms_ratio_median"] - 1.0) < 1e-12
    assert summary["rmse_median"] == 0.0
    assert summary["max_abs_diff_max"] == 0.0


def test_sensor_matrix_summary_transpose_alignment():
    module = _load_module()
    mat = np.arange(24, dtype=float).reshape(4, 6)
    summary = module.summarize_sensor_matrix_metrics(mat.T, mat, expected_sensors=4)
    assert summary["n_sensors"] == 4.0
    assert summary["n_time_samples"] == 6.0
    assert abs(summary["pearson_r_mean"] - 1.0) < 1e-12


def test_pml_outside_padding_shapes_and_values():
    module = _load_module()

    volume_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    padded_2d = module.pad_volume_for_pml_outside(volume_2d, (1, 2, 3))
    assert padded_2d.shape == (4, 6, 1)
    assert padded_2d[1:3, 2:4, 0].tolist() == volume_2d.tolist()
    assert np.count_nonzero(padded_2d) == volume_2d.size

    volume_3d = np.ones((2, 3, 4), dtype=float)
    padded_3d = module.pad_volume_for_pml_outside(volume_3d, (2, 1, 1))
    assert padded_3d.shape == (6, 5, 6)
    assert np.allclose(padded_3d[2:4, 1:4, 1:5], volume_3d)

    assert module.expand_pml_outside_shape((2, 3), (1, 2, 3)) == (4, 7, 1)
    assert module.expand_pml_outside_shape((2, 3, 4), (1, 2, 3)) == (4, 7, 10)


def test_pml_outside_padding_rejects_2d_tuple_for_3d_volume():
    module = _load_module()

    volume_3d = np.ones((2, 3, 4), dtype=float)
    try:
        module.pad_volume_for_pml_outside(volume_3d, (1, 2))
    except ValueError as exc:
        assert "3-D PML tuple" in str(exc)
    else:
        raise AssertionError("Expected pad_volume_for_pml_outside to reject a 2-D PML tuple")
