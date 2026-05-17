"""Controlled Chapter 29 CT placement rendering helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

CT_FRAME_KEYS = (
    "common_target",
    "linear_exposure",
    "linear_active",
    "linear_fusion",
    "nonlinear_pressure",
    "nonlinear_fwi",
    "nonlinear_cavitation_source",
    "nonlinear_cavitation",
    "nonlinear_fusion",
    "fusion_difference",
)

def placement_fields(linear: dict[str, object]) -> dict[str, np.ndarray]:
    ct = np.asarray(linear["placement_ct_hu"], dtype=float)
    extent = np.asarray(_image_extent_xy(ct, np.asarray(linear["placement_spacing_m"], dtype=float)))
    therapy = np.asarray(linear["placement_therapy_points_m"], dtype=float)[:, :2]
    imaging = np.asarray(linear["placement_imaging_points_m"], dtype=float)[:, :2]
    return {
        "placement_ct_hu": ct,
        "placement_extent_m": extent,
        "placement_target_mask": np.asarray(linear["placement_target_mask"], dtype=bool),
        "placement_body_mask": np.asarray(linear["placement_body_mask"], dtype=bool),
        "placement_therapy_points_xy_m": therapy,
        "placement_imaging_points_xy_m": imaging,
        "placement_focus_m": np.asarray(linear["placement_focus_m"], dtype=float)[:2],
        "placement_skin_contact_m": np.asarray(linear["placement_skin_contact_m"], dtype=float)[:2],
        "placement_xlim_m": np.asarray(_axis_limits(extent[:2], therapy[:, 0], imaging[:, 0])),
        "placement_ylim_m": np.asarray(_axis_limits(extent[2:], therapy[:, 1], imaging[:, 1])),
    }

def ct_frame_key(key: str) -> str:
    return f"ct_frame_{key}"

def add_ct_frame_fields(fields: dict[str, np.ndarray], source_extent: list[float]) -> None:
    _ensure_axis_fields(fields)
    target_shape = np.asarray(fields["placement_ct_hu"]).shape
    target_extent = [float(v) for v in np.asarray(fields["placement_extent_m"], dtype=float)]
    for key in CT_FRAME_KEYS:
        image = _resample_to_extent(np.asarray(fields[key], dtype=float), source_extent, target_shape, target_extent)
        fields[ct_frame_key(key)] = image >= 0.5 if key == "common_target" else image

def plot_placement_context(ax: plt.Axes, comparison: dict[str, object]) -> plt.AxesImage:
    fields = comparison["fields"]
    _ensure_axis_fields(fields)
    ct = np.asarray(fields["placement_ct_hu"], dtype=float)
    extent = [float(v) for v in np.asarray(fields["placement_extent_m"], dtype=float)]
    therapy = np.asarray(fields["placement_therapy_points_xy_m"], dtype=float)
    imaging = np.asarray(fields["placement_imaging_points_xy_m"], dtype=float)
    focus = np.asarray(fields["placement_focus_m"], dtype=float)
    skin = np.asarray(fields["placement_skin_contact_m"], dtype=float)
    target_key = ct_frame_key("common_target") if ct_frame_key("common_target") in fields else "placement_target_mask"
    im = ax.imshow(ct.T, cmap="gray", origin="lower", extent=extent, vmin=-200, vmax=300)
    _contour_mask(ax, np.asarray(fields[target_key], dtype=bool), extent, "white", 1.0)
    _contour_mask(ax, np.asarray(fields["placement_body_mask"], dtype=bool), extent, "cyan", 0.7)
    if therapy.size:
        ax.scatter(therapy[:, 0], therapy[:, 1], s=2.0, c="#e74c3c", alpha=0.50)
    if imaging.size:
        ax.scatter(imaging[:, 0], imaging[:, 1], s=5.0, c="#2e86de", alpha=0.80)
    ax.scatter([focus[0]], [focus[1]], marker="x", s=34, c="white", linewidths=1.3)
    ax.scatter([skin[0]], [skin[1]], marker="o", s=18, c="lime", edgecolors="black", linewidths=0.4)
    apply_placement_axes(ax, fields)
    return im

def apply_placement_axes(ax: plt.Axes, fields: dict[str, np.ndarray]) -> None:
    _ensure_axis_fields(fields)
    ax.set_xlim(*[float(v) for v in np.asarray(fields["placement_xlim_m"], dtype=float)])
    ax.set_ylim(*[float(v) for v in np.asarray(fields["placement_ylim_m"], dtype=float)])

def _ensure_axis_fields(fields: dict[str, np.ndarray]) -> None:
    if "placement_xlim_m" in fields and "placement_ylim_m" in fields:
        return
    extent = [float(v) for v in np.asarray(fields["placement_extent_m"], dtype=float)]
    therapy = np.asarray(fields["placement_therapy_points_xy_m"], dtype=float)
    imaging = np.asarray(fields["placement_imaging_points_xy_m"], dtype=float)
    fields["placement_xlim_m"] = np.asarray(
        _axis_limits(extent[:2], therapy[:, 0] if therapy.size else [], imaging[:, 0] if imaging.size else [])
    )
    fields["placement_ylim_m"] = np.asarray(
        _axis_limits(extent[2:], therapy[:, 1] if therapy.size else [], imaging[:, 1] if imaging.size else [])
    )

def _resample_to_extent(
    image: np.ndarray,
    source_extent: list[float],
    target_shape: tuple[int, int],
    target_extent: list[float],
) -> np.ndarray:
    sx, sy = image.shape
    tx, ty = target_shape
    x = np.linspace(target_extent[0], target_extent[1], tx)
    y = np.linspace(target_extent[2], target_extent[3], ty)
    u = (x - source_extent[0]) * (sx - 1) / max(source_extent[1] - source_extent[0], 1.0e-12)
    v = (y - source_extent[2]) * (sy - 1) / max(source_extent[3] - source_extent[2], 1.0e-12)
    out = np.zeros((tx, ty), dtype=float)
    valid_x = (u >= 0.0) & (u <= sx - 1)
    valid_y = (v >= 0.0) & (v <= sy - 1)
    for ix, ux in enumerate(u):
        if not valid_x[ix]:
            continue
        x0 = int(np.floor(ux))
        x1 = min(x0 + 1, sx - 1)
        wx = ux - x0
        for iy, vy in enumerate(v):
            if not valid_y[iy]:
                continue
            y0 = int(np.floor(vy))
            y1 = min(y0 + 1, sy - 1)
            wy = vy - y0
            out[ix, iy] = (
                (1.0 - wx) * (1.0 - wy) * image[x0, y0]
                + wx * (1.0 - wy) * image[x1, y0]
                + (1.0 - wx) * wy * image[x0, y1]
                + wx * wy * image[x1, y1]
            )
    return out

def _image_extent_xy(image: np.ndarray, spacing_m: np.ndarray) -> list[float]:
    nx, ny = image.shape
    return [
        -0.5 * (nx - 1) * float(spacing_m[0]),
        0.5 * (nx - 1) * float(spacing_m[0]),
        -0.5 * (ny - 1) * float(spacing_m[1]),
        0.5 * (ny - 1) * float(spacing_m[1]),
    ]

def _contour_mask(ax: plt.Axes, mask: np.ndarray, extent: list[float], color: str, width: float) -> None:
    x = np.linspace(extent[0], extent[1], mask.shape[0])
    y = np.linspace(extent[2], extent[3], mask.shape[1])
    ax.contour(x, y, mask.T.astype(float), levels=[0.5], colors=color, linewidths=width)

def _axis_limits(base: list[float], *points: object) -> tuple[float, float]:
    values = [float(base[0]), float(base[1])]
    for point_set in points:
        arr = np.asarray(point_set, dtype=float).ravel()
        if arr.size:
            values.extend([float(np.min(arr)), float(np.max(arr))])
    low, high = min(values), max(values)
    pad = max(0.04 * (high - low), 1.0e-3)
    return low - pad, high + pad
