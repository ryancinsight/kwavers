"""
Chapter 25: Transcranial Brain Focused Ultrasound Planning
==========================================================

Runnable planning example for:

1. CT, T1 MRI, and MNI152 atlas ingestion.
2. RITK-based CT/MRI/atlas registration when the compiled RITK Python
   extension is installed.
3. 1024-element hemispherical focused-bowl array construction.
4. Skull path phase correction, acoustic field synthesis, Pennes thermal dose,
   and cavitation-risk mapping for essential-tremor VIM ablation.
5. CT-space segmentation or GBM segmentation hooks for BBB-opening subspots.

The default local run uses compact NIfTI volumes already present under
``data/rire_patient_109``, ``data/niivue``, and
``data/mni_icbm152_2009c``. BBB planning executes from CT plus segmentation
when ``data/ct_segmentation_sample/segmentation.nii.gz`` is present, or from a
CT-backed CFB-GBM case when available.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BOOK_DIR = Path(__file__).resolve().parent
if str(BOOK_DIR) not in sys.path:
    sys.path.insert(0, str(BOOK_DIR))

from transcranial_planning.data import (  # noqa: E402
    FIG_DIR,
    LOCAL_RIRE_CT,
    dataset_sources,
    discover_gbm_case,
    load_default_brain_triplet,
    load_gbm_case,
    load_nifti,
)
from transcranial_planning.figures import (  # noqa: E402
    dataset_manifest,
    plot_affine_registration_qc,
    plot_gbm_bbb_opening,
    plot_essential_tremor_result,
    plot_gbm_plan,
    plot_registration_inputs,
    plot_transducer_phase,
    write_json,
)
from transcranial_planning.registration import (  # noqa: E402
    affine_register_moving_to_fixed,
    register_triplet_with_ritk,
)
from transcranial_planning.modality_bridge import modality_bridge_manifest  # noqa: E402
from transcranial_planning.scene import CANONICAL_BRAIN_SCENE  # noqa: E402
from transcranial_planning.simulation import (  # noqa: E402
    acoustic_observables,
    bbb_opening_from_subspots,
    gbm_subspot_plan,
    pennes_thermal_dose,
    rayleigh_pressure_field,
)
from transcranial_planning.transducer import (  # noqa: E402
    TransducerConfig,
    phase_correction_through_ct,
)


def run() -> dict:
    print("[ch25] Loading local CT/MRI/MNI volumes")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    scene = CANONICAL_BRAIN_SCENE
    triplet = load_default_brain_triplet(shape=(48, 60, 48), scene=scene)
    sources = dataset_sources()
    write_json("dataset_manifest.json", dataset_manifest(sources))

    print("[ch25] Running RITK registration adapter")
    registration = register_triplet_with_ritk(triplet, max_iterations=12)
    plot_registration_inputs(triplet, registration)

    print("[ch25] Building 1024-element transducer and skull phase correction")
    transducer = TransducerConfig.from_scene(scene)
    phase = phase_correction_through_ct(
        triplet.ct_hu.data,
        triplet.ct_hu.spacing_m,
        triplet.target_index,
        transducer,
        samples_per_ray=192,
        skull_mask=triplet.skull_mask,
    )
    plot_transducer_phase(triplet, phase)

    print("[ch25] Synthesizing acoustic field and thermal dose")
    pressure = rayleigh_pressure_field(
        phase,
        triplet.ct_hu.data.shape,
        triplet.ct_hu.spacing_m,
        triplet.target_index,
        transducer,
        target_peak_pressure_pa=1.0e6,
        chunk_points=768,
    )
    acoustic = acoustic_observables(pressure, transducer.frequency_hz)
    thermal = pennes_thermal_dose(
        acoustic.intensity_w_m2,
        triplet.skull_mask,
        triplet.brain_mask,
        triplet.ct_hu.spacing_m,
        sonication_s=12.0,
        dt_s=0.25,
    )
    plot_essential_tremor_result(triplet, acoustic, thermal)

    print("[ch25] Checking optional GBM CT/MRI/segmentation case")
    gbm_paths = discover_gbm_case()
    bridge_manifest = modality_bridge_manifest(gbm_paths)
    write_json("modality_bridge_manifest.json", bridge_manifest)
    gbm = load_gbm_case(shape=triplet.ct_hu.data.shape, paths=gbm_paths)
    gbm_payload: dict[str, object]
    if gbm is None:
        gbm_payload = {
            "executed": False,
            "reason": (
                "No GBM case found. Expected CFB-GBM under data/cfb_gbm_sample "
                "or UPenn-GBM NIfTI files under data/upenn_gbm_sample."
            ),
            "modality_bridge": bridge_manifest["plan"],
        }
        print(f"  GBM path not executed: {gbm_payload['reason']}")
    else:
        ct = gbm.ct
        tumor = gbm.tumor
        affine_ct_payload: dict[str, object] | None = None
        if ct is None and gbm.segmentation_space == "mri" and LOCAL_RIRE_CT.exists():
            print("[ch25] Affine-registering sample CT to GBM MRI for visual QC")
            sample_ct = load_nifti(LOCAL_RIRE_CT, "RIRE sample CT")
            affine_ct = affine_register_moving_to_fixed(
                sample_ct,
                gbm.planning_reference,
                "Affine-registered sample CT in GBM MRI space",
            )
            plot_affine_registration_qc(affine_ct, tumor)
            ct = affine_ct.moving_registered
            affine_ct_payload = {
                "executed": affine_ct.executed,
                "method": affine_ct.method,
                "message": affine_ct.message,
                "source_ct": str(LOCAL_RIRE_CT),
                "nmi": float(affine_ct.nmi),
                "edge_overlap": float(affine_ct.edge_overlap),
                "visual_qc": str(FIG_DIR / "fig06_affine_ct_to_mri_qc.png"),
            }
        planning_volume = ct if ct is not None and gbm.segmentation_space == "ct" else gbm.planning_reference
        plan = gbm_subspot_plan(tumor, planning_volume.spacing_m, pitch_m=3.0e-3)
        plot_gbm_plan(tumor, plan)
        bbb = bbb_opening_from_subspots(
            tumor,
            plan,
            planning_volume.spacing_m,
            mechanical_index=0.45,
            sonication_s=60.0,
            duty_cycle=0.02,
            focal_radius_m=2.0e-3,
        )
        plot_gbm_bbb_opening(tumor, bbb)
        gbm_payload = {
            "executed": True,
            "dataset": gbm.dataset,
            "segmentation_space": gbm.segmentation_space,
            "ct": str(ct.source_path) if ct is not None else None,
            "t1": str(gbm.t1.source_path) if gbm.t1 is not None else None,
            "t1gd": str(gbm.t1gd.source_path) if gbm.t1gd is not None else None,
            "flair": str(gbm.flair.source_path) if gbm.flair is not None else None,
            "t2": str(gbm.t2.source_path) if gbm.t2 is not None else None,
            "available_modalities": list(gbm.available_modalities),
            "planning_reference_modality": gbm.planning_reference_modality,
            "subspots": int(plan.indices.shape[0]),
            "covered_fraction": float(plan.covered_fraction),
            "bbb_opened_fraction": float(np.count_nonzero(bbb.opened_mask & tumor) / np.count_nonzero(tumor)),
            "peak_bbb_permeability": float(bbb.permeability.max()),
            "peak_inertial_cavitation_risk": float(bbb.inertial_cavitation_risk.max()),
            "ct_backed": bool(ct is not None and gbm.segmentation_space == "ct"),
            "affine_sample_ct_to_mri": affine_ct_payload,
            "modality_bridge": bridge_manifest["plan"],
        }

    metrics = {
        "registration": {
            "executed": registration.executed,
            "method": registration.method,
            "message": registration.message,
            "ncc_t1_before": registration.ncc_t1_before,
            "ncc_t1_after": registration.ncc_t1_after,
            "nmi_t1_before": registration.nmi_t1_before,
            "nmi_t1_after": registration.nmi_t1_after,
            "mse_t1_before": registration.mse_t1_before,
            "mse_t1_after": registration.mse_t1_after,
            "ncc_atlas_before": registration.ncc_atlas_before,
            "ncc_atlas_after": registration.ncc_atlas_after,
            "nmi_atlas_before": registration.nmi_atlas_before,
            "nmi_atlas_after": registration.nmi_atlas_after,
            "mse_atlas_before": registration.mse_atlas_before,
            "mse_atlas_after": registration.mse_atlas_after,
        },
        "scene": scene.to_manifest(),
        "transducer": {
            "elements": transducer.element_count,
            "frequency_hz": transducer.frequency_hz,
            "radius_m": transducer.radius_m,
            "phase_span_rad": float(phase.phases_rad.max() - phase.phases_rad.min()),
            "mean_skull_path_m": float(phase.skull_lengths_m.mean()),
            "mean_amplitude_weight": float(phase.amplitude_weights.mean()),
            "min_amplitude_weight": float(phase.amplitude_weights.min()),
        },
        "essential_tremor": {
            "target_mni_mm": triplet.target_world_mm,
            "target_index": triplet.target_index,
            "peak_pressure_pa": float(acoustic.pressure_pa.max()),
            "peak_mi": float(acoustic.mechanical_index.max()),
            "peak_temperature_c": float(thermal.peak_temperature_c.max()),
            "max_cem43_min": float(thermal.cem43_min.max()),
            "lesion_voxels": int(thermal.lesion_mask.sum()),
            "max_cavitation_probability": float(acoustic.cavitation_probability.max()),
        },
        "gbm": gbm_payload,
    }
    write_json("metrics.json", metrics)
    print("[ch25] Complete")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


if __name__ == "__main__" or __name__.startswith("ch25"):
    run()
