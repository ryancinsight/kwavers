"""CT slice preparation for abdominal FWI."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import ndimage

from abdominal_fwi.model import PreparedSlice


def prepare_ct_slice(
    *,
    name: str,
    title: str,
    ct_hu: np.ndarray,
    label: np.ndarray,
    sound_speed_m_s: np.ndarray,
    focus_index: int,
    input_spacing_m: float,
    organ_labels: Iterable[int],
    target_labels: Iterable[int],
    focus_indices: tuple[int, int, int] | None = None,
    slice_axis: int = 0,
    output_size: int = 72,
    margin_voxels: int = 44,
) -> PreparedSlice:
    """Extract a tumor-centered CT slice and resample it to a fixed grid."""

    focus = focus_indices or (int(focus_index), label.shape[1] // 2, label.shape[2] // 2)
    label_slice = centered_slice(label, focus, slice_axis)
    target_mask = np.isin(label_slice, tuple(target_labels))
    organ_mask = np.isin(label_slice, tuple(organ_labels)) | target_mask
    if not np.any(target_mask):
        raise ValueError(f"{name}: selected focus slice has no target mask")

    y_idx, z_idx = np.where(target_mask)
    side = max(
        int(y_idx.max()) - int(y_idx.min()) + 1,
        int(z_idx.max()) - int(z_idx.min()) + 1,
    ) + 2 * margin_voxels
    side = min(side, label_slice.shape[0], label_slice.shape[1])
    y0, y1 = clamped_window(int(round(float(np.mean(y_idx)))), side, label_slice.shape[0])
    z0, z1 = clamped_window(int(round(float(np.mean(z_idx)))), side, label_slice.shape[1])

    ct_slice = centered_slice(ct_hu, focus, slice_axis)
    c_slice = centered_slice(sound_speed_m_s, focus, slice_axis)
    ct_crop = np.asarray(ct_slice[y0:y1, z0:z1], dtype=np.float32)
    c_crop = np.asarray(c_slice[y0:y1, z0:z1], dtype=np.float32)
    label_crop = np.asarray(label_slice[y0:y1, z0:z1], dtype=np.int16)
    organ_crop = organ_mask[y0:y1, z0:z1]
    target_crop = target_mask[y0:y1, z0:z1]

    zoom = (output_size / ct_crop.shape[0], output_size / ct_crop.shape[1])
    ct_out = ndimage.zoom(ct_crop, zoom, order=1).astype(np.float32)
    c_out = ndimage.zoom(c_crop, zoom, order=1).astype(np.float32)
    label_out = ndimage.zoom(label_crop, zoom, order=0).astype(np.int16)
    organ_out = ndimage.zoom(organ_crop.astype(np.float32), zoom, order=0) > 0.5
    target_out = ndimage.zoom(target_crop.astype(np.float32), zoom, order=0) > 0.5
    imaging_out = imaging_support(ct_out, label_out, organ_out, target_out)
    c_out = ct_textured_sound_speed(ct_out, c_out, imaging_out, organ_out, target_out)

    return PreparedSlice(
        name=name,
        title=title,
        ct_hu=ct_out,
        sound_speed_m_s=c_out,
        label=label_out,
        imaging_mask=imaging_out,
        organ_mask=organ_out,
        target_mask=target_out,
        spacing_m=float(input_spacing_m / zoom[0]),
        source_index=int(focus[slice_axis]),
    )


def centered_slice(
    volume: np.ndarray,
    focus_indices: tuple[int, int, int],
    axis: int,
) -> np.ndarray:
    """Return one tumor-centered anatomical plane."""

    return np.asarray(np.take(volume, int(focus_indices[axis]), axis=axis))


def clamped_window(center: int, side: int, limit: int) -> tuple[int, int]:
    """Return a fixed-length window centered as closely as bounds permit."""

    start = center - side // 2
    start = max(0, min(start, limit - side))
    return start, start + side


def imaging_support(
    ct_hu: np.ndarray,
    label: np.ndarray,
    organ_mask: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    """Define the anatomical region inverted by FWI."""

    body = (ct_hu > -450.0) | (label > 0) | organ_mask | target_mask
    body = ndimage.binary_closing(body, structure=np.ones((3, 3), dtype=bool))
    body = ndimage.binary_fill_holes(body)
    return body | organ_mask | target_mask


def ct_textured_sound_speed(
    ct_hu: np.ndarray,
    label_speed_m_s: np.ndarray,
    imaging_mask: np.ndarray,
    organ_mask: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    """Inject CT-derived anatomical texture into the acoustic model."""

    ct = ndimage.gaussian_filter(ct_hu.astype(np.float32), sigma=0.75)
    speed = label_speed_m_s.astype(np.float32).copy()
    active = imaging_mask & (ct > -450.0)
    if np.any(active):
        centered = np.clip(ct[active], -150.0, 250.0)
        centered = centered - float(np.median(centered))
        speed[active] = speed[active] + 0.28 * centered.astype(np.float32)
    organ_like = (organ_mask | target_mask) & imaging_mask
    if np.any(organ_like):
        texture = np.clip(ct[organ_like], -75.0, 175.0)
        texture = texture - float(np.median(texture))
        speed[organ_like] = speed[organ_like] + 0.55 * texture.astype(np.float32)
    bone = imaging_mask & (ct > 250.0)
    if np.any(bone):
        speed[bone] = 2450.0 + 0.45 * np.clip(ct[bone] - 250.0, 0.0, 1200.0)
    return speed
