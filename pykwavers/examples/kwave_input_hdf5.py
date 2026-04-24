"""k-Wave save-to-disk HDF5 parity helpers.

This module mirrors the upstream k-Wave Python save-to-disk contract for the
subset used by the `na_controlling_the_pml` parity example:

* grid and PML scalars
* homogeneous medium scalars
* initial-pressure source input
* binary sensor-mask indices
* root-file metadata

The helpers write the same semantic datasets and attributes as the vendored
`k-wave-python` path, then compare two input files by dataset values and
attributes.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np

import kwave
from kwave.utils.data import get_date_string


_ROOT_ATTR_ORDER = (
    "created_by",
    "creation_date",
    "file_description",
    "file_type",
    "major_version",
    "minor_version",
)

_FLOAT_DATASET_ORDER = (
    "c0",
    "c_ref",
    "dt",
    "dx",
    "dy",
    "dz",
    "pml_x_alpha",
    "pml_y_alpha",
    "pml_z_alpha",
    "rho0",
    "rho0_sgx",
    "rho0_sgy",
    "rho0_sgz",
    "p0_source_input",
)

_INTEGER_DATASET_ORDER = (
    "Nt",
    "Nx",
    "Ny",
    "Nz",
    "absorbing_flag",
    "axisymmetric_flag",
    "elastic_flag",
    "nonlinear_flag",
    "nonuniform_grid_flag",
    "p0_source_flag",
    "p_source_flag",
    "pml_x_size",
    "pml_y_size",
    "pml_z_size",
    "sxx_source_flag",
    "sxy_source_flag",
    "sxz_source_flag",
    "syy_source_flag",
    "syz_source_flag",
    "szz_source_flag",
    "transducer_source_flag",
    "ux_source_flag",
    "uy_source_flag",
    "uz_source_flag",
    "sensor_mask_type",
    "sensor_mask_index",
)


def _normalize_triplet(values: tuple[int, ...] | tuple[float, ...], *, default_z: int | float) -> tuple:
    if len(values) == 2:
        return values[0], values[1], default_z
    if len(values) == 3:
        return values
    raise ValueError(f"Expected a 2- or 3-tuple, got {values!r}")


def _coerce_attr_value(value: object) -> str | bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, np.bytes_):
        return bytes(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"Unsupported HDF5 attribute value type: {type(value)!r}")


def build_root_attributes(
    *,
    file_description: str | bytes | np.bytes_ | None = None,
    creation_date: str | bytes | np.bytes_ | None = None,
    created_by: str | bytes | np.bytes_ | None = None,
    file_type: str | bytes | np.bytes_ = "input",
) -> dict[str, str | bytes]:
    """Build the root-file metadata used by k-Wave input files."""
    user_name = os.environ.get("USERNAME") or os.environ.get("USER") or "unknown"
    default_description = f"Input data created by {user_name} running MATLAB N/A on {platform.system()}"
    return {
        "created_by": _coerce_attr_value(created_by or f"k-Wave {kwave.__version__}"),
        "creation_date": _coerce_attr_value(creation_date or get_date_string()),
        "file_description": _coerce_attr_value(file_description or default_description),
        "file_type": _coerce_attr_value(file_type),
        "major_version": "1",
        "minor_version": "2",
    }


def _kwave_matrix_layout(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix)
    if arr.ndim == 0:
        return arr.reshape((1, 1, 1))
    if arr.ndim == 1:
        arr = arr.reshape((-1, 1))
        arr = arr.T
        return arr.reshape((1, arr.shape[0], arr.shape[1]))
    if arr.ndim == 2:
        arr = arr.T
        return arr.reshape((1, arr.shape[0], arr.shape[1]))
    if arr.ndim == 3:
        return np.transpose(arr, (2, 1, 0))
    raise ValueError(f"Unsupported matrix rank {arr.ndim}; expected 0-3 dimensions")


def _write_dataset(
    handle: h5py.File,
    name: str,
    value: np.ndarray | int | float,
    *,
    data_type: np.dtype,
    data_type_attr: str,
) -> None:
    laid_out = _kwave_matrix_layout(np.asarray(value, dtype=data_type))
    dataset = handle.create_dataset(name, data=laid_out, dtype=data_type)
    dataset.attrs.create("data_type", data_type_attr, None, dtype=f"<S{len(data_type_attr)}")
    dataset.attrs.create("domain_type", "real", None, dtype="<S4")


def _write_root_attributes(handle: h5py.File, root_attrs: Mapping[str, str | bytes]) -> None:
    for key in _ROOT_ATTR_ORDER:
        value = root_attrs[key]
        value_bytes = _coerce_attr_value(value)
        handle.attrs.create(key, value_bytes, None, dtype=f"<S{len(value_bytes)}")


def write_kwave_input_file(
    path: Path | str,
    *,
    grid_shape: tuple[int, int, int],
    grid_spacing: tuple[float, float, float],
    nt: int,
    dt: float,
    pml_size: tuple[int, ...],
    pml_alpha: tuple[float, ...],
    c0: float,
    c_ref: float,
    rho0: float,
    source_p0: np.ndarray,
    sensor_mask: np.ndarray,
    root_attrs: Mapping[str, str | bytes] | None = None,
) -> Path:
    """Write a k-Wave-style HDF5 input file.

    The writer mirrors the save-to-disk contract used by the vendored
    `na_controlling_the_pml` example. It is intentionally semantic rather than
    byte-for-byte identical: dataset names, shapes, dtypes, and attributes match
    the reference file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    nx, ny, nz = (int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]))
    px, py, pz = _normalize_triplet(tuple(int(v) for v in pml_size), default_z=0)
    ax, ay, az = _normalize_triplet(tuple(float(v) for v in pml_alpha), default_z=0.0)
    dx, dy, dz = (float(grid_spacing[0]), float(grid_spacing[1]), float(grid_spacing[2]))

    float_vars: dict[str, np.ndarray | float] = {
        "c0": float(c0),
        "c_ref": float(c_ref),
        "dt": float(dt),
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "pml_x_alpha": float(ax),
        "pml_y_alpha": float(ay),
        "pml_z_alpha": float(az),
        "rho0": float(rho0),
        "rho0_sgx": float(rho0),
        "rho0_sgy": float(rho0),
        "rho0_sgz": float(rho0),
        "p0_source_input": np.asarray(source_p0, dtype=np.float32),
    }

    integer_vars: dict[str, np.ndarray | int] = {
        "Nt": int(nt),
        "Nx": nx,
        "Ny": ny,
        "Nz": nz,
        "absorbing_flag": 0,
        "axisymmetric_flag": 0,
        "elastic_flag": 0,
        "nonlinear_flag": 0,
        "nonuniform_grid_flag": 0,
        "p0_source_flag": 1,
        "p_source_flag": 0,
        "pml_x_size": px,
        "pml_y_size": py,
        "pml_z_size": pz,
        "sxx_source_flag": 0,
        "sxy_source_flag": 0,
        "sxz_source_flag": 0,
        "syy_source_flag": 0,
        "syz_source_flag": 0,
        "szz_source_flag": 0,
        "transducer_source_flag": 0,
        "ux_source_flag": 0,
        "uy_source_flag": 0,
        "uz_source_flag": 0,
        "sensor_mask_type": 0,
        "sensor_mask_index": (np.flatnonzero(np.asarray(sensor_mask, dtype=bool).flatten(order="F")) + 1).reshape(-1, 1),
    }

    if nz == 1:
        for key in ("dz", "pml_z_alpha", "rho0_sgz"):
            float_vars.pop(key, None)

    with h5py.File(path, "w") as handle:
        for name in _FLOAT_DATASET_ORDER:
            if name in float_vars:
                _write_dataset(handle, name, float_vars[name], data_type=np.float32, data_type_attr="float")

        for name in _INTEGER_DATASET_ORDER:
            if name in integer_vars:
                _write_dataset(handle, name, integer_vars[name], data_type=np.uint64, data_type_attr="long")

        _write_root_attributes(handle, root_attrs or build_root_attributes())

    return path


def _attr_value_equal(lhs: object, rhs: object) -> bool:
    if isinstance(lhs, np.ndarray) or isinstance(rhs, np.ndarray):
        return np.array_equal(np.asarray(lhs), np.asarray(rhs))
    if isinstance(lhs, np.bytes_):
        lhs = bytes(lhs)
    if isinstance(rhs, np.bytes_):
        rhs = bytes(rhs)
    return lhs == rhs


def compare_kwave_input_files(reference_path: Path | str, candidate_path: Path | str) -> dict[str, object]:
    """Compare two k-Wave input HDF5 files with exact semantic equality."""
    reference_path = Path(reference_path)
    candidate_path = Path(candidate_path)

    with h5py.File(reference_path, "r") as reference, h5py.File(candidate_path, "r") as candidate:
        ref_keys = set(reference.keys())
        cand_keys = set(candidate.keys())
        missing = sorted(ref_keys - cand_keys)
        extra = sorted(cand_keys - ref_keys)

        dataset_results: dict[str, dict[str, object]] = {}
        attr_results: dict[str, dict[str, object]] = {}
        max_abs_diff = 0.0

        for name in sorted(ref_keys & cand_keys):
            ref_ds = reference[name]
            cand_ds = candidate[name]
            ref_value = ref_ds[...]
            cand_value = cand_ds[...]
            data_match = ref_value.shape == cand_value.shape and ref_value.dtype == cand_value.dtype and np.array_equal(
                ref_value, cand_value
            )
            if data_match:
                diff = 0.0
            else:
                diff = float(np.max(np.abs(ref_value.astype(np.float64) - cand_value.astype(np.float64))))
            max_abs_diff = max(max_abs_diff, diff)

            ref_attrs = dict(ref_ds.attrs.items())
            cand_attrs = dict(cand_ds.attrs.items())
            attr_names = sorted(set(ref_attrs) | set(cand_attrs))
            attr_mismatches: dict[str, dict[str, object]] = {}
            for attr_name in attr_names:
                if attr_name not in ref_attrs or attr_name not in cand_attrs:
                    attr_mismatches[attr_name] = {
                        "reference": ref_attrs.get(attr_name),
                        "candidate": cand_attrs.get(attr_name),
                    }
                    continue
                if not _attr_value_equal(ref_attrs[attr_name], cand_attrs[attr_name]):
                    attr_mismatches[attr_name] = {
                        "reference": ref_attrs[attr_name],
                        "candidate": cand_attrs[attr_name],
                    }

            dataset_results[name] = {
                "shape_match": ref_value.shape == cand_value.shape,
                "dtype_match": ref_value.dtype == cand_value.dtype,
                "data_match": data_match,
                "max_abs_diff": diff,
                "attrs_match": not attr_mismatches,
                "attr_mismatches": attr_mismatches,
            }
            attr_results[name] = attr_mismatches

        root_attr_names = sorted(set(reference.attrs.keys()) | set(candidate.attrs.keys()))
        root_attr_mismatches: dict[str, dict[str, object]] = {}
        for attr_name in root_attr_names:
            ref_value = reference.attrs.get(attr_name)
            cand_value = candidate.attrs.get(attr_name)
            if ref_value is None or cand_value is None:
                root_attr_mismatches[attr_name] = {"reference": ref_value, "candidate": cand_value}
                continue
            if not _attr_value_equal(ref_value, cand_value):
                root_attr_mismatches[attr_name] = {"reference": ref_value, "candidate": cand_value}

        status = "PASS" if not missing and not extra and not root_attr_mismatches and all(
            result["data_match"] and result["attrs_match"] for result in dataset_results.values()
        ) else "FAIL"

        return {
            "reference_path": str(reference_path),
            "candidate_path": str(candidate_path),
            "status": status,
            "missing_datasets": missing,
            "extra_datasets": extra,
            "root_attr_mismatches": root_attr_mismatches,
            "dataset_results": dataset_results,
            "dataset_attr_mismatches": attr_results,
            "max_abs_diff": max_abs_diff,
        }
