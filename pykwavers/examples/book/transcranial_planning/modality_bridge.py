from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .data import GbmCasePaths, LOCAL_RIRE_CT


@dataclass(frozen=True)
class BridgeReference:
    name: str
    url: str
    role: str
    boundary: str


@dataclass(frozen=True)
class BridgeAction:
    action: str
    status: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    rationale: str
    boundary: str
    reference: str | None = None
    artifact_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class PairingRequirement:
    name: str
    required_for: str
    satisfied: bool
    evidence: str
    boundary: str


@dataclass(frozen=True)
class ModalityBridgePlan:
    dataset: str | None
    planning_space: str
    simulation_scope: str
    simulation_ready: bool
    skull_acoustics_same_subject: bool
    available_modalities: tuple[str, ...]
    missing_modalities: tuple[str, ...]
    requirements: tuple[PairingRequirement, ...]
    actions: tuple[BridgeAction, ...]
    references: tuple[BridgeReference, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "planning_space": self.planning_space,
            "simulation_scope": self.simulation_scope,
            "simulation_ready": self.simulation_ready,
            "skull_acoustics_same_subject": self.skull_acoustics_same_subject,
            "available_modalities": list(self.available_modalities),
            "missing_modalities": list(self.missing_modalities),
            "requirements": [
                {
                    "name": requirement.name,
                    "required_for": requirement.required_for,
                    "satisfied": requirement.satisfied,
                    "evidence": requirement.evidence,
                    "boundary": requirement.boundary,
                }
                for requirement in self.requirements
            ],
            "actions": [
                {
                    "action": action.action,
                    "status": action.status,
                    "inputs": list(action.inputs),
                    "outputs": list(action.outputs),
                    "rationale": action.rationale,
                    "boundary": action.boundary,
                    "reference": action.reference,
                    "artifact_paths": list(action.artifact_paths),
                }
                for action in self.actions
            ],
            "references": [
                {
                    "name": reference.name,
                    "url": reference.url,
                    "role": reference.role,
                    "boundary": reference.boundary,
                }
                for reference in self.references
            ],
        }


def bridge_references() -> tuple[BridgeReference, ...]:
    return (
        BridgeReference(
            "cWDM",
            "https://huggingface.co/papers/2411.17203",
            "paired high-resolution 3D medical image translation; missing brain MRI modality synthesis from available modalities; CT-to-MR or MR-to-CT is treated as a paired-translation candidate only when a trained checkpoint and output NIfTI are supplied",
            "not executed by the Chapter 25 script; synthetic outputs must be generated outside this repo, registered to the planning lattice, and quality checked before use",
        ),
        BridgeReference(
            "SLaM-DiMM",
            "https://huggingface.co/papers/2509.16019",
            "missing T1w, T1ce, T2w, or FLAIR MRI synthesis from available brain MRI modalities",
            "MRI-only bridge; it does not provide skull CT acoustics and cannot make an MRI-space case CT-backed",
        ),
        BridgeReference(
            "NV-Segment-CTMR",
            "https://huggingface.co/nvidia/NV-Segment-CTMR",
            "research 3D CT/MR segmentation reference for anatomical prompt-based masks",
            "not a substitute for expert GBM segmentation unless target-label validation and local review are added",
        ),
    )


def build_modality_bridge_plan(
    paths: GbmCasePaths | None,
    sample_ct_path: Path = LOCAL_RIRE_CT,
) -> ModalityBridgePlan:
    if paths is None:
        references = bridge_references()
        return ModalityBridgePlan(
            dataset=None,
            planning_space="none",
            simulation_scope="not_executable",
            simulation_ready=False,
            skull_acoustics_same_subject=False,
            available_modalities=(),
            missing_modalities=("ct", "t1gd", "flair", "segmentation"),
            requirements=(
                PairingRequirement(
                    "ct_segmentation_pair",
                    "CT-space skull acoustics and BBB subspot simulation",
                    False,
                    "no GBM case discovered",
                    "simulation stops before GBM branch",
                ),
            ),
            actions=(
                BridgeAction(
                    "ingest_real_case",
                    "required",
                    (),
                    ("ct", "mri", "segmentation"),
                    "No real case manifest is available.",
                    "Place a CT-backed CFB-GBM case, a CT-space segmentation case, or an MRI-backed UPenn-GBM case under the documented data roots.",
                    None,
                ),
            ),
            references=references,
        )

    available = _available_modalities(paths)
    missing = _missing_modalities(paths)
    requirements = _pairing_requirements(paths)
    actions = list(_base_actions(paths, missing, sample_ct_path))
    actions.extend(_synthesis_actions(paths, available, missing))
    ct_space = paths.ct is not None and paths.segmentation_space == "ct"
    mri_space = paths.ct is None and paths.segmentation_space == "mri"
    return ModalityBridgePlan(
        dataset=paths.dataset,
        planning_space=paths.segmentation_space,
        simulation_scope=_simulation_scope(ct_space, mri_space),
        simulation_ready=ct_space or mri_space,
        skull_acoustics_same_subject=ct_space,
        available_modalities=available,
        missing_modalities=missing,
        requirements=requirements,
        actions=tuple(actions),
        references=bridge_references(),
    )


def modality_bridge_manifest(paths: GbmCasePaths | None) -> dict[str, object]:
    plan = build_modality_bridge_plan(paths)
    return {
        "contract": {
            "authoritative_planning_order": [
                "same-patient CT plus CT-space segmentation",
                "same-patient CT plus MRI-derived segmentation registered into CT space",
                "MRI-space segmentation with explicit non-CT-backed boundary",
            ],
            "simulation_rejection_rules": [
                "do not infer skull acoustics from MRI-only GBM data",
                "do not treat cross-subject sample CT as same-patient evidence",
                "do not use synthetic CT or MRI unless a real NIfTI output and QC metrics are present",
            ],
        },
        "plan": plan.to_dict(),
    }


def _available_modalities(paths: GbmCasePaths) -> tuple[str, ...]:
    modalities = []
    if paths.ct is not None:
        modalities.append("ct")
    if paths.t1 is not None:
        modalities.append("t1")
    if _is_real_mri_path(paths, paths.t1gd):
        modalities.append("t1gd")
    if _is_real_mri_path(paths, paths.flair):
        modalities.append("flair")
    if paths.t2 is not None:
        modalities.append("t2")
    modalities.append("segmentation")
    return tuple(modalities)


def _missing_modalities(paths: GbmCasePaths) -> tuple[str, ...]:
    missing = []
    if paths.ct is None:
        missing.append("ct")
    if paths.t1 is None:
        missing.append("t1")
    if not _is_real_mri_path(paths, paths.t1gd):
        missing.append("t1gd")
    if not _is_real_mri_path(paths, paths.flair):
        missing.append("flair")
    if paths.t2 is None:
        missing.append("t2")
    return tuple(missing)


def _pairing_requirements(paths: GbmCasePaths) -> tuple[PairingRequirement, ...]:
    ct_segmentation = paths.ct is not None and paths.segmentation_space == "ct"
    t1gd = paths.t1gd if _is_real_mri_path(paths, paths.t1gd) else None
    flair = paths.flair if _is_real_mri_path(paths, paths.flair) else None
    mri_segmentation = (
        paths.segmentation_space == "mri"
        and t1gd is not None
        and flair is not None
    )
    ct_mri = paths.ct is not None and (
        paths.t1 is not None or t1gd is not None or flair is not None
    )
    return (
        PairingRequirement(
            "ct_segmentation_pair",
            "skull phase correction, attenuation, reflection, and CT-space BBB dose",
            ct_segmentation,
            _evidence(paths.ct, paths.segmentation),
            "required for same-patient skull-acoustic GBM treatment simulation",
        ),
        PairingRequirement(
            "mri_segmentation_pair",
            "GBM target delineation and MRI-space subspot geometry",
            mri_segmentation,
            _evidence(t1gd, flair, paths.segmentation),
            "sufficient for tumor geometry but not for skull acoustics",
        ),
        PairingRequirement(
            "ct_mri_pair",
            "same-subject CT/MRI registration and CT-space tumor transfer",
            ct_mri,
            _evidence(paths.ct, paths.t1, t1gd, flair),
            "required before an MRI-derived tumor mask can become CT-backed",
        ),
    )


def _base_actions(
    paths: GbmCasePaths,
    missing: tuple[str, ...],
    sample_ct_path: Path,
) -> tuple[BridgeAction, ...]:
    if paths.ct is not None and paths.segmentation_space == "ct":
        return (
            BridgeAction(
                "accept_ct_space_segmentation",
                "ready",
                ("ct", "segmentation"),
                ("ct_space_tumor_mask",),
                "The segmentation is already defined on the CT planning lattice.",
                "Skull acoustics and BBB subspot dose can use the same CT-space voxel spacing.",
                None,
            ),
        )
    if "ct" in missing and paths.segmentation_space == "mri":
        sample_status = "available" if sample_ct_path.exists() else "unavailable"
        return (
            BridgeAction(
                "accept_mri_space_segmentation",
                "ready_with_boundary",
                ("t1gd", "flair", "segmentation"),
                ("mri_space_tumor_mask",),
                "The UPenn-style case supplies real co-registered MRI and expert tumor segmentation.",
                "BBB subspot geometry can execute in MRI space; skull phase correction remains outside the GBM branch until same-patient CT or accepted synthetic CT is supplied.",
                None,
            ),
            BridgeAction(
                "registered_sample_ct_visual_qc",
                sample_status,
                ("sample_ct", "t1gd"),
                ("ct_contour_overlay",),
                "A sample CT can be affine-resampled to the MRI lattice to inspect gross alignment behavior.",
                "This is visual QC only and never promotes an MRI-only GBM case to CT-backed skull acoustics.",
                None,
            ),
        )
    return (
        BridgeAction(
            "reject_incomplete_pairing",
            "blocked",
            _available_modalities(paths),
            (),
            "The discovered modality set does not satisfy a CT-space or MRI-space GBM planning contract.",
            "Add real paired CT/segmentation or MRI/segmentation before running the GBM branch.",
            None,
        ),
    )


def _synthesis_actions(
    paths: GbmCasePaths,
    available: tuple[str, ...],
    missing: tuple[str, ...],
) -> tuple[BridgeAction, ...]:
    actions: list[BridgeAction] = []
    if "ct" in missing:
        actions.append(
            BridgeAction(
                "external_synthetic_ct_candidate",
                "external_required",
                tuple(modality for modality in available if modality != "segmentation"),
                ("ct",),
                "cWDM is a recent paired-translation reference for high-resolution 3D medical image synthesis and is the cited route for MR-to-CT candidate generation.",
                "The Chapter 25 code does not synthesize CT; it only accepts a generated CT after it exists as a NIfTI artifact, is registered to the tumor lattice, and passes QC.",
                "cWDM",
                _artifact_paths(paths, ("synthetic_ct_cwdm.nii.gz", "synthetic_ct.nii.gz")),
            )
        )
    missing_mri = tuple(modality for modality in ("t1", "t1gd", "t2", "flair") if modality in missing)
    available_mri = tuple(modality for modality in ("t1", "t1gd", "t2", "flair") if modality in available)
    if missing_mri and len(available_mri) >= 2:
        actions.append(
            BridgeAction(
                "external_missing_mri_candidate",
                "external_required",
                available_mri,
                missing_mri,
                "SLaM-DiMM and cWDM provide recent references for missing brain MRI modality synthesis from available MRI modalities.",
                "Generated MRI is allowed only for completing MRI segmentation model inputs; it does not provide CT skull acoustics.",
                "SLaM-DiMM",
                _artifact_paths(paths, tuple(f"synthetic_{modality}_slam_dimm.nii.gz" for modality in missing_mri)),
            )
        )
    if paths.segmentation is None:
        actions.append(
            BridgeAction(
                "external_segmentation_candidate",
                "external_required",
                available,
                ("segmentation",),
                "NV-Segment-CTMR is a research reference for prompt-based 3D CT/MR segmentation.",
                "A generated mask must be target-label validated before it can replace expert GBM segmentation.",
                "NV-Segment-CTMR",
                _artifact_paths(paths, ("segmentation.nii.gz", "segmentation_nv_segment_ctmr.nii.gz")),
            )
        )
    return tuple(actions)


def _simulation_scope(ct_space: bool, mri_space: bool) -> str:
    if ct_space:
        return "ct_space_skull_acoustics_and_bbb_subspot_simulation"
    if mri_space:
        return "mri_space_gbm_subspot_geometry_without_same_patient_skull_acoustics"
    return "blocked_incomplete_modalities"


def _evidence(*paths: Path | None) -> str:
    present = [str(path) for path in paths if path is not None]
    return "; ".join(present) if present else "missing"


def _is_real_mri_path(paths: GbmCasePaths, candidate: Path | None) -> bool:
    if candidate is None:
        return False
    if paths.ct is not None and candidate == paths.ct:
        return False
    return True


def _artifact_paths(paths: GbmCasePaths, names: tuple[str, ...]) -> tuple[str, ...]:
    bridge_dir = paths.segmentation.parent / "bridge"
    return tuple(str(bridge_dir / name) for name in names)
