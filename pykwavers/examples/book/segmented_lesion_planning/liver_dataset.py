"""LiTS liver CT adapter for Chapter 32 transducer planning."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .types import SegmentationGrid, Tissue


NIFTI_DTYPES: dict[int, str] = {
    2: "u1",
    4: "i2",
    8: "i4",
    16: "f4",
    64: "f8",
    256: "i1",
    512: "u2",
    768: "u4",
}


def load_lits_liver_planning_grid(
    ct_path: Path,
    seg_path: Path,
    *,
    output_size: int = 112,
    margin_m: float = 0.060,
) -> tuple[SegmentationGrid, dict[str, object]]:
    """Load LiTS17 CT and liver/tumor labels into the Chapter 32 2-D contract.

    LiTS provides native labels for liver parenchyma (`1`) and tumor (`2`).
    Air, fat, and bone are derived from CT Hounsfield-unit thresholds.  The
    target is the largest connected tumor component on the selected slice; other
    label-2 tumor foci are treated as off-target avoid structures.  The avoid
    channel also includes a deterministic vascular-risk surrogate: enhanced
    voxels inside liver parenchyma with `160 < HU < 400`, excluding tumor.
    """

    if output_size < 48:
        raise ValueError("output_size must be at least 48")
    if margin_m <= 0.0:
        raise ValueError("margin_m must be positive")

    ct, ct_spacing_mm = load_nifti_array(ct_path)
    seg, seg_spacing_mm = load_nifti_array(seg_path)
    if ct.shape != seg.shape:
        raise ValueError(f"CT shape {ct.shape} does not match segmentation shape {seg.shape}")
    if len(ct.shape) != 3:
        raise ValueError("LiTS CT and segmentation must be 3-D")
    if not np.allclose(ct_spacing_mm[:3], seg_spacing_mm[:3], rtol=0.0, atol=1e-6):
        raise ValueError("CT and segmentation voxel spacings do not match")

    seg_i = np.asarray(seg, dtype=np.int16)
    tumor_counts = np.count_nonzero(seg_i == 2, axis=(0, 1))
    if int(np.max(tumor_counts)) == 0:
        raise ValueError("LiTS segmentation has no label-2 tumor voxels")

    slice_index = int(np.argmax(tumor_counts))
    ct_slice = np.asarray(ct[:, :, slice_index], dtype=float)
    seg_slice = np.asarray(seg_i[:, :, slice_index], dtype=np.int16)
    spacing_m = 0.5 * (float(ct_spacing_mm[0]) + float(ct_spacing_mm[1])) * 1.0e-3

    tumor_mask = seg_slice == 2
    crop = square_crop_bounds(tumor_mask, spacing_m, margin_m, output_size)
    ct_crop = ct_slice[crop["i0"] : crop["i1"], crop["j0"] : crop["j1"]]
    seg_crop = seg_slice[crop["i0"] : crop["i1"], crop["j0"] : crop["j1"]]
    ct_small = resize_nearest(ct_crop, output_size)
    seg_small = resize_nearest(seg_crop, output_size).astype(np.int16)

    labels, body_mask = classify_liver_slice(ct_small, seg_small)
    grid = SegmentationGrid(
        labels=labels,
        body_mask=body_mask,
        spacing_m=float(spacing_m * float(crop["width_vox"]) / float(output_size)),
    )
    metadata: dict[str, object] = {
        "source": "LiTS17 sample liver CT",
        "ct_path": str(ct_path),
        "segmentation_path": str(seg_path),
        "slice_index": slice_index,
        "native_shape": tuple(int(v) for v in ct.shape),
        "native_spacing_mm": tuple(float(v) for v in ct_spacing_mm[:3]),
        "crop_bounds": dict(crop),
        "output_size": int(output_size),
        "spacing_m": float(grid.spacing_m),
        "segmentation_labels": {"normal": 1, "tumor": 2},
        "target_rule": "largest connected label-2 component on selected slice",
        "derived_labels": {
            "air": "HU < -700 or outside body mask",
            "fat": "-500 <= HU < -100 outside liver/tumor",
            "bone": "HU >= 200 outside liver/tumor",
            "avoid": "160 < HU < 400 inside liver label, excluding tumor",
        },
    }
    return grid, metadata


def load_nifti_array(path: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    """Read a single-file NIfTI-1 array without optional Python dependencies."""

    raw = read_binary(path)
    header = raw[:348]
    endian = nifti_endian(header)
    dim = struct.unpack(endian + "8h", header[40:56])
    ndim = int(dim[0])
    shape = tuple(int(value) for value in dim[1 : 1 + ndim])
    datatype = int(struct.unpack(endian + "h", header[70:72])[0])
    if datatype not in NIFTI_DTYPES:
        raise ValueError(f"unsupported NIfTI datatype code {datatype}")

    pixdim = struct.unpack(endian + "8f", header[76:108])
    offset = int(struct.unpack(endian + "f", header[108:112])[0])
    slope = float(struct.unpack(endian + "f", header[112:116])[0])
    intercept = float(struct.unpack(endian + "f", header[116:120])[0])
    dtype = np.dtype(endian + NIFTI_DTYPES[datatype])
    count = int(np.prod(shape))
    data = np.frombuffer(raw, dtype=dtype, count=count, offset=offset).reshape(shape, order="F")
    array = np.asarray(data)
    if slope != 0.0 and (slope != 1.0 or intercept != 0.0):
        array = array.astype(np.float32) * slope + intercept
    return array, tuple(float(value) for value in pixdim[1 : 1 + ndim])


def read_binary(path: Path) -> bytes:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return handle.read()
    with path.open("rb") as handle:
        return read_all(handle)


def read_all(handle: BinaryIO) -> bytes:
    return handle.read()


def nifti_endian(header: bytes) -> str:
    if len(header) < 348:
        raise ValueError("NIfTI file is shorter than the 348-byte header")
    if struct.unpack("<i", header[:4])[0] == 348:
        return "<"
    if struct.unpack(">i", header[:4])[0] == 348:
        return ">"
    raise ValueError("file is not a NIfTI-1 single-file image")


def square_crop_bounds(
    tumor_mask: np.ndarray,
    spacing_m: float,
    margin_m: float,
    output_size: int,
) -> dict[str, int]:
    coords = np.argwhere(tumor_mask)
    if coords.size == 0:
        raise ValueError("tumor mask is empty")
    low = coords.min(axis=0)
    high = coords.max(axis=0) + 1
    center = 0.5 * (low + high)
    tumor_width = int(max(high - low))
    margin_vox = int(np.ceil(margin_m / spacing_m))
    width = max(int(output_size), tumor_width + 2 * margin_vox)
    width = min(width + (width % 2), int(min(tumor_mask.shape)))
    start = np.rint(center - 0.5 * width).astype(int)
    upper = np.asarray(tumor_mask.shape, dtype=int) - width
    start = np.maximum(0, np.minimum(start, upper))
    stop = start + width
    return {
        "i0": int(start[0]),
        "i1": int(stop[0]),
        "j0": int(start[1]),
        "j1": int(stop[1]),
        "width_vox": int(width),
    }


def resize_nearest(array: np.ndarray, output_size: int) -> np.ndarray:
    rows = np.linspace(0, array.shape[0] - 1, output_size).round().astype(int)
    cols = np.linspace(0, array.shape[1] - 1, output_size).round().astype(int)
    return array[np.ix_(rows, cols)]


def classify_liver_slice(ct_hu: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if ct_hu.shape != seg.shape:
        raise ValueError("CT slice and segmentation slice must have the same shape")
    labels = np.full(seg.shape, int(Tissue.AIR), dtype=np.uint8)
    body = (ct_hu > -500.0) | (seg > 0)
    labels[body] = int(Tissue.NORMAL)

    non_liver = body & (seg == 0)
    labels[non_liver & (ct_hu >= -500.0) & (ct_hu < -100.0)] = int(Tissue.FAT)
    labels[non_liver & (ct_hu >= 200.0)] = int(Tissue.BONE)
    labels[(ct_hu < -700.0) & (seg == 0)] = int(Tissue.AIR)

    liver = seg == 1
    all_tumor = seg == 2
    tumor = largest_connected_component(all_tumor)
    untreated_tumor = all_tumor & ~tumor
    vascular_avoid = liver & (ct_hu > 160.0) & (ct_hu < 400.0)
    labels[liver] = int(Tissue.NORMAL)
    labels[vascular_avoid] = int(Tissue.AVOID)
    labels[untreated_tumor] = int(Tissue.AVOID)
    labels[tumor] = int(Tissue.TUMOR)
    body_mask = body & (labels != int(Tissue.AIR))
    return labels, body_mask


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask.copy()
    visited = np.zeros(mask.shape, dtype=bool)
    best: list[tuple[int, int]] = []
    for start in np.argwhere(mask):
        i0, j0 = int(start[0]), int(start[1])
        if visited[i0, j0]:
            continue
        component: list[tuple[int, int]] = []
        stack = [(i0, j0)]
        visited[i0, j0] = True
        while stack:
            i, j = stack.pop()
            component.append((i, j))
            for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1] and mask[ni, nj] and not visited[ni, nj]:
                    visited[ni, nj] = True
                    stack.append((ni, nj))
        if len(component) > len(best):
            best = component
    output = np.zeros(mask.shape, dtype=bool)
    for i, j in best:
        output[i, j] = True
    return output
