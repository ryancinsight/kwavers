from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, zoom

from .scene import CANONICAL_BRAIN_SCENE, BrainSceneDefinition


REPO_ROOT = Path(__file__).resolve().parents[4]
FIG_DIR = REPO_ROOT / "docs" / "book" / "figures" / "ch25"

LOCAL_CT = REPO_ROOT / "data" / "niivue" / "CT_Philips.nii.gz"
LOCAL_RIRE_CT = REPO_ROOT / "data" / "rire_patient_109" / "patient_109_ct.nii.gz"
LOCAL_RIRE_T1 = REPO_ROOT / "data" / "rire_patient_109" / "patient_109_mr_t1.nii.gz"
LOCAL_RIRE_T2 = REPO_ROOT / "data" / "rire_patient_109" / "patient_109_mr_t2.nii.gz"
LOCAL_T1 = REPO_ROOT / "data" / "niivue" / "chris_t1.nii.gz"
LOCAL_T2 = REPO_ROOT / "data" / "niivue" / "chris_t2.nii.gz"
LOCAL_MNI = (
    REPO_ROOT
    / "data"
    / "mni_icbm152_2009c"
    / "mni_icbm152_nlin_sym_09c"
    / "mni_icbm152_t1_tal_nlin_sym_09c.nii"
)
LOCAL_MNI_MASK = (
    REPO_ROOT
    / "data"
    / "mni_icbm152_2009c"
    / "mni_icbm152_nlin_sym_09c"
    / "mni_icbm152_t1_tal_nlin_sym_09c_mask.nii"
)
GBM_ROOT = REPO_ROOT / "data" / "cfb_gbm_sample"
UPENN_GBM_ROOT = REPO_ROOT / "data" / "upenn_gbm_sample"
CT_SEGMENTATION_ROOT = REPO_ROOT / "data" / "ct_segmentation_sample"


@dataclass(frozen=True)
class DatasetSource:
    name: str
    role: str
    path: Path | None
    source_url: str
    license: str
    present: bool


@dataclass(frozen=True)
class Volume:
    name: str
    data: np.ndarray
    spacing_m: tuple[float, float, float]
    affine: np.ndarray
    source_path: Path


@dataclass(frozen=True)
class BrainTriplet:
    ct_hu: Volume
    t1: Volume
    atlas_t1: Volume
    atlas_mask: Volume
    skull_mask: np.ndarray
    brain_mask: np.ndarray
    target_index: tuple[int, int, int]
    target_world_mm: tuple[float, float, float]


@dataclass(frozen=True)
class GbmCasePaths:
    dataset: str
    ct: Path | None
    segmentation_space: str
    t1: Path | None
    t1gd: Path | None
    flair: Path | None
    t2: Path | None
    segmentation: Path


@dataclass(frozen=True)
class GbmLoadedCase:
    dataset: str
    segmentation_space: str
    ct: Volume | None
    t1: Volume | None
    t1gd: Volume | None
    flair: Volume | None
    t2: Volume | None
    segmentation: Volume
    tumor: np.ndarray
    planning_reference: Volume
    planning_reference_modality: str
    available_modalities: tuple[str, ...]


def dataset_sources() -> list[DatasetSource]:
    return [
        DatasetSource(
            "NiiVue cranial CT/T1/T2 sample",
            "runnable local CT/MRI default",
            LOCAL_CT,
            "https://github.com/niivue/niivue-images",
            "sample data distributed in repository",
            LOCAL_CT.exists() and LOCAL_T1.exists() and LOCAL_T2.exists(),
        ),
        DatasetSource(
            "RIRE patient 109 CT",
            "same-patient CT skull acoustic map",
            LOCAL_RIRE_CT,
            "https://rire.insight-journal.org/download_data",
            "CC BY 3.0 US",
            LOCAL_RIRE_CT.exists(),
        ),
        DatasetSource(
            "RIRE patient 109 MR-T1/T2",
            "same-patient MRI for CT/MR registration",
            LOCAL_RIRE_T1,
            "https://pyscience.wordpress.com/2014/11/02/multi-modal-image-segmentation-with-python-simpleitk/",
            "RIRE-derived mirror; source dataset CC BY 3.0 US",
            LOCAL_RIRE_T1.exists() and LOCAL_RIRE_T2.exists(),
        ),
        DatasetSource(
            "MNI ICBM152 nonlinear 2009c symmetric",
            "atlas and target coordinate frame",
            LOCAL_MNI,
            "https://www.mcgill.ca/bic/node/73",
            "MNI/McGill permissive atlas terms",
            LOCAL_MNI.exists() and LOCAL_MNI_MASK.exists(),
        ),
        DatasetSource(
            "CFB-GBM",
            "preferred same-patient CT/MRI/segmentation cohort",
            GBM_ROOT,
            "https://www.cancerimagingarchive.net/collection/cfb-gbm/",
            "CC BY 4.0",
            discover_cfb_gbm_case() is not None,
        ),
        DatasetSource(
            "RIRE CT-space segmentation sample",
            "executable CT+segmentation BBB planning contract",
            CT_SEGMENTATION_ROOT,
            "https://rire.insight-journal.org/download_data",
            "CT: CC BY 3.0 US; segmentation: deterministic local annotation",
            discover_ct_segmentation_case() is not None,
        ),
        DatasetSource(
            "UPenn-GBM subject sub-002",
            "local executable GBM MRI/segmentation sample",
            UPENN_GBM_ROOT,
            "https://github.com/data-nih/tcia/releases/tag/upenn-gbm",
            "CC BY 4.0",
            discover_upenn_gbm_case() is not None,
        ),
        DatasetSource(
            "GLIS-RT",
            "DICOM GBM/glioma CT/MR/RTSTRUCT fallback",
            None,
            "https://www.cancerimagingarchive.net/collection/glis-rt/",
            "NIH controlled access",
            False,
        ),
    ]


def discover_gbm_case(root: Path = GBM_ROOT) -> GbmCasePaths | None:
    return discover_cfb_gbm_case(root) or discover_ct_segmentation_case() or discover_upenn_gbm_case()


def discover_cfb_gbm_case(root: Path = GBM_ROOT) -> GbmCasePaths | None:
    if not root.is_dir():
        return None
    candidates = []
    for case_dir in [root, *[p for p in root.iterdir() if p.is_dir()]]:
        paths = {
            "dataset": "CFB-GBM",
            "ct": _first_existing(case_dir, ("ct.nii.gz", "CT.nii.gz")),
            "segmentation_space": "ct",
            "t1": _first_existing(case_dir, ("t1.nii.gz", "T1.nii.gz")),
            "t1gd": _first_existing(case_dir, ("t1gd.nii.gz", "t1ce.nii.gz", "T1Gd.nii.gz")),
            "flair": _first_existing(case_dir, ("flair.nii.gz", "FLAIR.nii.gz")),
            "t2": _first_existing(case_dir, ("t2.nii.gz", "T2.nii.gz")),
            "segmentation": _first_existing(case_dir, ("seg.nii.gz", "segmentation.nii.gz", "mask.nii.gz")),
        }
        if paths["ct"] is not None and paths["segmentation"] is not None:
            candidates.append(GbmCasePaths(**paths))
    return candidates[0] if candidates else None


def discover_ct_segmentation_case(root: Path = CT_SEGMENTATION_ROOT) -> GbmCasePaths | None:
    ct = LOCAL_RIRE_CT if LOCAL_RIRE_CT.exists() else None
    segmentation = root / "segmentation.nii.gz"
    if ct is None or not segmentation.exists():
        return None
    return GbmCasePaths(
        dataset="RIRE-CT-segmentation",
        ct=ct,
        segmentation_space="ct",
        t1=None,
        t1gd=None,
        flair=None,
        t2=None,
        segmentation=segmentation,
    )


def discover_upenn_gbm_case(root: Path = UPENN_GBM_ROOT) -> GbmCasePaths | None:
    if not root.is_dir():
        return None
    case_dirs = [root, *[p for p in root.iterdir() if p.is_dir()]]
    for case_dir in case_dirs:
        names = {p.name for p in case_dir.glob("*.nii.gz")}
        subject_prefixes = sorted({name.split("_", 1)[0] for name in names if name.startswith("sub-")})
        for prefix in subject_prefixes:
            paths = {
                "dataset": "UPenn-GBM",
                "ct": None,
                "segmentation_space": "mri",
                "t1": _first_existing(case_dir, (f"{prefix}_T1w.nii.gz",)),
                "t1gd": _first_existing(case_dir, (f"{prefix}_ce-gd_T1w.nii.gz",)),
                "flair": _first_existing(case_dir, (f"{prefix}_FLAIR.nii.gz",)),
                "t2": _first_existing(case_dir, (f"{prefix}_T2w.nii.gz",)),
                "segmentation": _first_existing(case_dir, (f"{prefix}_seg.nii.gz",)),
            }
            has_mri = any(paths[name] is not None for name in ("t1", "t1gd", "flair", "t2"))
            if has_mri and paths["segmentation"] is not None:
                return GbmCasePaths(**paths)
    return None


def _first_existing(case_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = case_dir / name
        if candidate.exists():
            return candidate
    return None


def load_nifti(path: Path, name: str) -> Volume:
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    spacing_m = tuple(float(v) * 1.0e-3 for v in image.header.get_zooms()[:3])
    return Volume(name, np.asarray(data, dtype=np.float32), spacing_m, image.affine, path)


def resample_volume(volume: Volume, shape: tuple[int, int, int], order: int) -> Volume:
    factors = tuple(n / old for n, old in zip(shape, volume.data.shape))
    data = zoom(volume.data, factors, order=order).astype(np.float32)
    spacing = tuple(s * old / new for s, old, new in zip(volume.spacing_m, volume.data.shape, shape))
    index_scale = np.diag([old / new for old, new in zip(volume.data.shape, shape)] + [1.0])
    affine = volume.affine @ index_scale
    return Volume(volume.name, data, spacing, affine, volume.source_path)


def normalize_unit(data: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = data[mask] if mask is not None and np.any(mask) else data
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.0))
    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def skull_mask_from_ct(ct_hu: np.ndarray) -> np.ndarray:
    skull = ct_hu > 300.0
    skull = binary_closing(skull, iterations=1)
    return skull.astype(bool)


def brain_mask_from_atlas(mask: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    atlas = zoom(mask > 0.5, tuple(n / old for n, old in zip(shape, mask.shape)), order=0)
    return binary_fill_holes(atlas > 0.5).astype(bool)


def brain_mask_from_ct(ct_hu: np.ndarray) -> np.ndarray:
    skull = skull_mask_from_ct(ct_hu)
    intracranial = np.zeros_like(skull, dtype=bool)
    for z in range(skull.shape[2]):
        closed_skull = binary_closing(skull[:, :, z], iterations=1)
        filled_skull = binary_fill_holes(closed_skull)
        if np.count_nonzero(filled_skull) > np.count_nonzero(closed_skull):
            intracranial[:, :, z] = filled_skull
    brain = binary_closing(intracranial & ~skull, iterations=1)
    if np.any(brain):
        return brain.astype(bool)
    head = binary_closing(ct_hu > -300.0, iterations=1)
    head = binary_fill_holes(head)
    return head.astype(bool)


def load_default_brain_triplet(
    shape: tuple[int, int, int] = (64, 80, 64),
    scene: BrainSceneDefinition = CANONICAL_BRAIN_SCENE,
) -> BrainTriplet:
    ct_path = LOCAL_RIRE_CT if LOCAL_RIRE_CT.exists() else LOCAL_CT
    t1_path = LOCAL_RIRE_T1 if LOCAL_RIRE_T1.exists() else LOCAL_T1
    missing = [p for p in (ct_path, t1_path, LOCAL_MNI, LOCAL_MNI_MASK) if not p.exists()]
    if missing:
        joined = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Required local brain data missing: {joined}")

    ct = resample_volume(load_nifti(ct_path, "RIRE CT" if ct_path == LOCAL_RIRE_CT else "CT Philips"), shape, order=1)
    t1 = resample_volume(load_nifti(t1_path, "RIRE MR-T1" if t1_path == LOCAL_RIRE_T1 else "T1 MRI"), shape, order=1)
    atlas = resample_volume(load_nifti(LOCAL_MNI, "MNI152 2009c T1"), shape, order=1)
    atlas_mask_raw = load_nifti(LOCAL_MNI_MASK, "MNI152 brain mask")
    atlas_mask = resample_volume(atlas_mask_raw, shape, order=0)
    brain_mask = brain_mask_from_ct(ct.data)
    target_index = scene.target.resolve_index(brain_mask)

    return BrainTriplet(
        ct_hu=ct,
        t1=t1,
        atlas_t1=atlas,
        atlas_mask=atlas_mask,
        skull_mask=skull_mask_from_ct(ct.data),
        brain_mask=brain_mask,
        target_index=target_index,
        target_world_mm=scene.target.source_world_mm,
    )


def world_to_resampled_index(
    affine: np.ndarray,
    original_shape: tuple[int, int, int],
    resampled_shape: tuple[int, int, int],
    world_mm: np.ndarray,
) -> tuple[int, int, int]:
    original_index = nib.affines.apply_affine(np.linalg.inv(affine), world_mm)
    scaled = original_index * (np.asarray(resampled_shape) - 1.0) / (np.asarray(original_shape) - 1.0)
    clipped = np.clip(np.rint(scaled), 0, np.asarray(resampled_shape) - 1).astype(int)
    return tuple(int(v) for v in clipped)


def atlas_world_to_target_mask_index(
    atlas_affine: np.ndarray,
    atlas_mask: np.ndarray,
    target_mask: np.ndarray,
    world_mm: np.ndarray,
) -> tuple[int, int, int]:
    atlas_coords = np.argwhere(atlas_mask)
    target_coords = np.argwhere(target_mask)
    if atlas_coords.size == 0 or target_coords.size == 0:
        raise ValueError("Atlas and target masks must contain foreground voxels")
    atlas_lo = atlas_coords.min(axis=0).astype(np.float64)
    atlas_hi = atlas_coords.max(axis=0).astype(np.float64)
    target_lo = target_coords.min(axis=0).astype(np.float64)
    target_hi = target_coords.max(axis=0).astype(np.float64)
    atlas_index = nib.affines.apply_affine(np.linalg.inv(atlas_affine), world_mm)
    normalized = (atlas_index - atlas_lo) / np.maximum(atlas_hi - atlas_lo, 1.0)
    clipped = np.clip(target_lo + normalized * (target_hi - target_lo), target_lo, target_hi)
    return tuple(int(v) for v in np.rint(clipped).astype(int))


def load_gbm_case(
    shape: tuple[int, int, int] = (64, 80, 64),
    paths: GbmCasePaths | None = None,
) -> GbmLoadedCase | None:
    paths = paths or discover_gbm_case()
    if paths is None:
        return None
    ct = resample_volume(load_nifti(paths.ct, f"{paths.dataset} CT"), shape, order=1) if paths.ct is not None else None
    t1 = _load_optional_volume(paths.t1, f"{paths.dataset} T1", shape)
    t1gd = _load_optional_volume(paths.t1gd, f"{paths.dataset} T1Gd", shape)
    flair = _load_optional_volume(paths.flair, f"{paths.dataset} FLAIR", shape)
    t2 = _load_optional_volume(paths.t2, f"{paths.dataset} T2", shape)
    seg = resample_volume(load_nifti(paths.segmentation, f"{paths.dataset} segmentation"), shape, order=0)
    tumor = seg.data > 0.5
    if not np.any(tumor):
        raise ValueError(f"GBM segmentation contains no foreground voxels: {paths.segmentation}")
    planning_modality, planning_reference = _planning_reference_volume(paths.segmentation_space, ct, t1, t1gd, flair, t2)
    return GbmLoadedCase(
        dataset=paths.dataset,
        segmentation_space=paths.segmentation_space,
        ct=ct,
        t1=t1,
        t1gd=t1gd,
        flair=flair,
        t2=t2,
        segmentation=seg,
        tumor=tumor,
        planning_reference=planning_reference,
        planning_reference_modality=planning_modality,
        available_modalities=_loaded_modalities(ct, t1, t1gd, flair, t2),
    )


def _load_optional_volume(path: Path | None, name: str, shape: tuple[int, int, int]) -> Volume | None:
    if path is None:
        return None
    return resample_volume(load_nifti(path, name), shape, order=1)


def _planning_reference_volume(
    segmentation_space: str,
    ct: Volume | None,
    t1: Volume | None,
    t1gd: Volume | None,
    flair: Volume | None,
    t2: Volume | None,
) -> tuple[str, Volume]:
    if segmentation_space == "ct":
        if ct is None:
            raise ValueError("CT-space segmentation requires a real CT volume")
        return "ct", ct
    for modality, volume in (("flair", flair), ("t1gd", t1gd), ("t2", t2), ("t1", t1)):
        if volume is not None:
            return modality, volume
    raise ValueError("MRI-space segmentation requires at least one real MRI volume")


def _loaded_modalities(
    ct: Volume | None,
    t1: Volume | None,
    t1gd: Volume | None,
    flair: Volume | None,
    t2: Volume | None,
) -> tuple[str, ...]:
    modalities = []
    for name, volume in (("ct", ct), ("t1", t1), ("t1gd", t1gd), ("flair", flair), ("t2", t2)):
        if volume is not None:
            modalities.append(name)
    modalities.append("segmentation")
    return tuple(modalities)
