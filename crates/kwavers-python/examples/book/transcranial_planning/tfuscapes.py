from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

from .benchmark import run_skull_adaptive_benchmark, summarize_benchmark_result
from .scene import CANONICAL_BRAIN_SCENE, BrainSceneDefinition


TFUSCAPES_DATASET_ID = "vinkle-srivastav/TFUScapes"
TFUSCAPES_REVISION = "1c410548e40c491cedd779648257a1c9eaee3587"
TFUSCAPES_DEFAULT_SHA256 = "3be28a4454251583ea161b0f1fcbc3df960a45cc481141eed61346df42d6e20e"
TFUSCAPES_BODY_HU_THRESHOLD = 1.0e-6
TFUSCAPES_SKULL_HU_THRESHOLD = CANONICAL_BRAIN_SCENE.transducer.skull_hu_threshold


@dataclass(frozen=True)
class TFUScapesCaseSpec:
    dataset_id: str
    revision: str
    split: str
    manifest_row: int
    manifest_text: str
    repo_path: str
    sha256: str

    @property
    def resolve_url(self) -> str:
        return f"https://huggingface.co/datasets/{self.dataset_id}/resolve/{self.revision}/{self.repo_path}"

    @property
    def cache_name(self) -> str:
        subject, filename = self.manifest_text.split("/", 1)
        return f"{subject}_{filename}"


DEFAULT_TFUSCAPES_CASE = TFUScapesCaseSpec(
    dataset_id=TFUSCAPES_DATASET_ID,
    revision=TFUSCAPES_REVISION,
    split="train",
    manifest_row=0,
    manifest_text="A00028185/exp_0.npz",
    repo_path="data/A00028185/exp_0.npz",
    sha256=TFUSCAPES_DEFAULT_SHA256,
)


@dataclass(frozen=True)
class TFUScapesCase:
    spec: TFUScapesCaseSpec
    path: Path
    ct_hu: np.ndarray
    pressure_pa: np.ndarray
    transducer_indices: np.ndarray

    @property
    def target_index(self) -> tuple[int, int, int]:
        index = np.unravel_index(int(np.nanargmax(self.pressure_pa)), self.pressure_pa.shape)
        return tuple(int(value) for value in index)

    @property
    def skull_mask(self) -> np.ndarray:
        return self.ct_hu >= TFUSCAPES_SKULL_HU_THRESHOLD

    @property
    def brain_mask(self) -> np.ndarray:
        return (self.ct_hu > TFUSCAPES_BODY_HU_THRESHOLD) & (self.ct_hu < TFUSCAPES_SKULL_HU_THRESHOLD)


@dataclass(frozen=True)
class TFUScapesSceneMapping:
    target_index: tuple[int, int, int]
    target_fraction_xyz: tuple[float, float, float]
    spacing_m: tuple[float, float, float]
    transducer_points_m: np.ndarray
    transducer_radius_vox_median: float
    transducer_radius_m_mean: float
    transducer_radius_m_std: float
    transducer_axis_unit: tuple[float, float, float]
    cap_angle_min_deg: float
    cap_angle_max_deg: float


def default_cache_dir() -> Path:
    env = os.environ.get("KWAVERS_TFUSCAPES_CACHE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "kwavers" / "tfuscapes"


def download_tfuscapes_case(
    spec: TFUScapesCaseSpec = DEFAULT_TFUSCAPES_CASE,
    cache_dir: Path | None = None,
) -> Path:
    cache = default_cache_dir() if cache_dir is None else cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / spec.cache_name
    if target.exists():
        verify_sha256(target, spec.sha256)
        return target

    partial = target.with_suffix(target.suffix + ".part")
    request = Request(spec.resolve_url, headers={"User-Agent": "kwavers-tfuscapes-import"})
    digest = hashlib.sha256()
    with urlopen(request, timeout=120) as response, partial.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            out.write(chunk)
    actual = digest.hexdigest()
    if actual != spec.sha256:
        partial.unlink(missing_ok=True)
        raise ValueError(f"TFUScapes case hash mismatch: expected {spec.sha256}, got {actual}")
    partial.replace(target)
    return target


def verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"TFUScapes case hash mismatch: expected {expected}, got {actual}")


def load_tfuscapes_case(
    path: Path | None = None,
    *,
    spec: TFUScapesCaseSpec = DEFAULT_TFUSCAPES_CASE,
    cache_dir: Path | None = None,
) -> TFUScapesCase:
    npz_path = download_tfuscapes_case(spec, cache_dir) if path is None else path
    if spec.sha256 and npz_path.name == spec.cache_name:
        verify_sha256(npz_path, spec.sha256)
    with np.load(npz_path) as payload:
        missing = {"ct", "pmap", "tr_coords"} - set(payload.files)
        if missing:
            raise ValueError(f"TFUScapes case missing required fields: {sorted(missing)}")
        ct = np.asarray(payload["ct"], dtype=np.float32)
        pressure = np.asarray(payload["pmap"], dtype=np.float32)
        transducer = np.asarray(payload["tr_coords"], dtype=np.float64)
    validate_tfuscapes_arrays(ct, pressure, transducer)
    return TFUScapesCase(spec, npz_path, ct, pressure, transducer)


def validate_tfuscapes_arrays(ct: np.ndarray, pressure: np.ndarray, transducer: np.ndarray) -> None:
    if ct.ndim != 3 or pressure.ndim != 3 or ct.shape != pressure.shape:
        raise ValueError("TFUScapes ct and pmap must be matching 3-D arrays")
    if transducer.ndim != 2 or transducer.shape[1] != 3 or transducer.shape[0] == 0:
        raise ValueError("TFUScapes tr_coords must have shape (N, 3) with N > 0")
    for name, array in (("ct", ct), ("pmap", pressure), ("tr_coords", transducer)):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"TFUScapes field contains non-finite values: {name}")
    if float(np.max(pressure)) <= 0.0:
        raise ValueError("TFUScapes pmap must contain a positive pressure peak")


def map_case_to_scene(
    case: TFUScapesCase,
    scene: BrainSceneDefinition = CANONICAL_BRAIN_SCENE,
) -> TFUScapesSceneMapping:
    brain = case.brain_mask
    if not np.any(brain):
        raise ValueError("TFUScapes brain support is empty after pseudo-CT thresholding")
    target = np.asarray(case.target_index, dtype=np.float64)
    if not brain[case.target_index]:
        raise ValueError("TFUScapes pressure peak target is outside the CT-derived brain support")

    coords = np.argwhere(brain)
    lo = coords.min(axis=0).astype(np.float64)
    hi = coords.max(axis=0).astype(np.float64)
    denom = np.maximum(hi - lo, 1.0)
    fraction = tuple(float(v) for v in np.clip((target - lo) / denom, 0.0, 1.0))

    relative_vox = case.transducer_indices - target[None, :]
    radii_vox = np.linalg.norm(relative_vox, axis=1)
    positive = radii_vox > 0.0
    if not np.any(positive):
        raise ValueError("TFUScapes transducer coordinates collapse onto the target")
    radius_vox = float(np.median(radii_vox[positive]))
    spacing = float(scene.transducer.radius_m / radius_vox)
    points_m = relative_vox * spacing
    radii_m = np.linalg.norm(points_m, axis=1)
    unit = points_m[positive] / radii_m[positive, None]
    axis = unit.mean(axis=0)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0:
        raise ValueError("TFUScapes transducer axis is undefined")
    axis = axis / axis_norm
    polar = np.rad2deg(np.arccos(np.clip(unit @ axis, -1.0, 1.0)))

    return TFUScapesSceneMapping(
        target_index=case.target_index,
        target_fraction_xyz=fraction,
        spacing_m=(spacing, spacing, spacing),
        transducer_points_m=points_m.astype(np.float64),
        transducer_radius_vox_median=radius_vox,
        transducer_radius_m_mean=float(np.mean(radii_m)),
        transducer_radius_m_std=float(np.std(radii_m)),
        transducer_axis_unit=tuple(float(v) for v in axis),
        cap_angle_min_deg=float(np.min(polar)),
        cap_angle_max_deg=float(np.max(polar)),
    )


def write_tfuscapes_ct_nifti(case: TFUScapesCase, mapping: TFUScapesSceneMapping, path: Path) -> Path:
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    spacing_mm = [value * 1.0e3 for value in mapping.spacing_m]
    affine = np.diag([spacing_mm[0], spacing_mm[1], spacing_mm[2], 1.0])
    image = nib.Nifti1Image(case.ct_hu.astype(np.float32), affine)
    image.header.set_zooms(tuple(spacing_mm))
    nib.save(image, str(path))
    return path


def compare_tfuscapes_case_to_scene(
    case: TFUScapesCase,
    mapping: TFUScapesSceneMapping,
    scene: BrainSceneDefinition = CANONICAL_BRAIN_SCENE,
) -> dict[str, Any]:
    return {
        "dataset": {
            "id": case.spec.dataset_id,
            "revision": case.spec.revision,
            "split": case.spec.split,
            "manifest_row": case.spec.manifest_row,
            "manifest_text": case.spec.manifest_text,
            "sha256": case.spec.sha256,
            "source_path": str(case.path),
        },
        "fields": {
            "ct": array_summary(case.ct_hu),
            "pmap": array_summary(case.pressure_pa),
            "tr_coords": array_summary(case.transducer_indices),
        },
        "target": {
            "index": list(mapping.target_index),
            "fraction_xyz": list(mapping.target_fraction_xyz),
            "pressure_peak_pa": float(np.max(case.pressure_pa)),
            "ct_hu_at_target": float(case.ct_hu[mapping.target_index]),
        },
        "transducer_geometry": {
            "coordinate_count": int(case.transducer_indices.shape[0]),
            "scene_element_count": int(scene.transducer.element_count),
            "scene_radius_m": float(scene.transducer.radius_m),
            "derived_spacing_m": float(mapping.spacing_m[0]),
            "median_radius_vox": mapping.transducer_radius_vox_median,
            "mean_radius_m": mapping.transducer_radius_m_mean,
            "radius_std_m": mapping.transducer_radius_m_std,
            "axis_unit": list(mapping.transducer_axis_unit),
            "cap_angle_min_deg": mapping.cap_angle_min_deg,
            "cap_angle_max_deg": mapping.cap_angle_max_deg,
        },
        "acceptance": {
            "minimal_fields_present": True,
            "target_from_pressure_peak_inside_brain_support": bool(case.brain_mask[mapping.target_index]),
            "ct_and_pressure_shapes_match": case.ct_hu.shape == case.pressure_pa.shape,
            "transducer_coordinates_are_index_space": True,
        },
    }


def array_summary(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def run_tfuscapes_skull_adaptive_benchmark(
    path: Path | None = None,
    *,
    spec: TFUScapesCaseSpec = DEFAULT_TFUSCAPES_CASE,
    cache_dir: Path | None = None,
    work_dir: Path | None = None,
    scene: BrainSceneDefinition = CANONICAL_BRAIN_SCENE,
    grid_size: int = 32,
    element_count: int | None = None,
    runner: Callable[..., dict[str, Any]] = run_skull_adaptive_benchmark,
) -> dict[str, Any]:
    case = load_tfuscapes_case(path, spec=spec, cache_dir=cache_dir)
    mapping = map_case_to_scene(case, scene)
    work = default_cache_dir() if work_dir is None else work_dir
    ct_nifti = write_tfuscapes_ct_nifti(case, mapping, work / f"{case.path.stem}_ct.nii.gz")
    raw_result = runner(
        ct_nifti,
        scene=scene,
        grid_size=grid_size,
        element_count=element_count,
        kwargs_overrides={
            "target_fraction_xyz": mapping.target_fraction_xyz,
            "body_hu_threshold": TFUSCAPES_BODY_HU_THRESHOLD,
            "skull_hu_threshold": TFUSCAPES_SKULL_HU_THRESHOLD,
        },
        summarize=False,
    )
    benchmark_summary = summarize_benchmark_result(raw_result)
    return {
        "case": compare_tfuscapes_case_to_scene(case, mapping, scene),
        "benchmark": benchmark_summary,
        "output_comparison": structural_output_comparison(case, raw_result),
    }


def structural_output_comparison(case: TFUScapesCase, result: dict[str, Any]) -> dict[str, Any]:
    reference = np.asarray(result["reference_pressure_pa"], dtype=np.float32)
    baseline = np.asarray(result["baseline_pressure_pa"], dtype=np.float32)
    return {
        "paper_pmap_shape": [int(v) for v in case.pressure_pa.shape],
        "kwavers_reference_shape": [int(v) for v in reference.shape],
        "kwavers_baseline_shape": [int(v) for v in baseline.shape],
        "paper_peak_index": list(case.target_index),
        "kwavers_focus_index": [int(v) for v in result["focus_index"]],
        "paper_peak_pressure_pa": float(np.max(case.pressure_pa)),
        "kwavers_reference_peak_pa": float(np.max(reference)),
        "kwavers_baseline_peak_pa": float(np.max(baseline)),
        "all_fields_are_rank_3": case.pressure_pa.ndim == reference.ndim == baseline.ndim == 3,
        "all_fields_are_finite": bool(
            np.all(np.isfinite(case.pressure_pa))
            and np.all(np.isfinite(reference))
            and np.all(np.isfinite(baseline))
        ),
    }
