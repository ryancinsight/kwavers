from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def load_example_module():
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    module_path = examples_dir / "us_bmode_phased_array_compare.py"
    sys.path.insert(0, str(examples_dir))
    spec = importlib.util.spec_from_file_location("us_bmode_phased_array_compare", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_active_voxel_orders_differ_as_expected():
    mod = load_example_module()

    class DummyTransducer:
        active_elements_mask = np.array(
            [
                [[0, 1], [1, 0], [0, 1]],
                [[1, 0], [0, 1], [1, 0]],
            ],
            dtype=np.uint8,
        )

    matlab_coords = mod.matlab_ordered_active_voxel_coords(DummyTransducer())
    c_coords = mod.c_order_active_voxel_coords(DummyTransducer())

    assert matlab_coords.shape == c_coords.shape
    assert set(map(tuple, matlab_coords.tolist())) == set(map(tuple, c_coords.tolist()))
    assert not np.array_equal(matlab_coords, c_coords)


def test_sensor_reorder_maps_c_order_rows_back_to_matlab_order():
    mod = load_example_module()

    class DummyTransducer:
        active_elements_mask = np.array(
            [
                [[0, 1], [1, 0], [0, 1]],
                [[1, 0], [0, 1], [1, 0]],
            ],
            dtype=np.uint8,
        )

    matlab_coords = mod.matlab_ordered_active_voxel_coords(DummyTransducer())
    c_coords = mod.c_order_active_voxel_coords(DummyTransducer())
    c_index = {tuple(coord.tolist()): idx for idx, coord in enumerate(c_coords)}
    sensor_rows = np.array([[10.0 * c_index[tuple(coord.tolist())] + 1.0] for coord in c_coords])

    reordered = mod.reorder_sensor_data_to_kwave_transducer_order(sensor_rows, DummyTransducer())
    expected = np.array([[10.0 * c_index[tuple(coord.tolist())] + 1.0] for coord in matlab_coords])

    assert np.array_equal(reordered, expected)
