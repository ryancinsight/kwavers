//! PyO3 binding for the transcranial FUS planning pipeline.
//!
//! Loads a CT NIfTI volume (and optional segmentation), derives tissue masks,
//! and delegates all physics to `run_transcranial_fus_planning`.

use kwavers::clinical::therapy::theranostic_guidance::{
    run_transcranial_fus_planning, target_index_from_mask_fraction_3d, TranscranialFusPlanConfig,
};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

use super::helpers::kwavers_to_py;
use crate::ritk_image::load_ritk_nifti;

/// Run the complete transcranial FUS therapy planning pipeline from a CT NIfTI
/// file.
///
/// Returns a dict with all planning fields as numpy arrays.
#[pyfunction]
#[pyo3(signature = (
    ct_nifti_path,
    segmentation_nifti_path = None,
    element_count = 1024,
    frequency_hz = 650_000.0,
    radius_m = 0.150,
    cap_min_polar_rad = 0.22,
    cap_max_polar_rad = 1.18,
    brain_sound_speed = 1540.0,
    skull_sound_speed = 2800.0,
    target_peak_pa = 1_000_000.0,
    samples_per_ray = 192,
    skull_hu_threshold = 300.0,
    body_hu_threshold = -350.0,
    target_fraction_xyz = None,
    mechanical_index_bbb = 0.45,
    sonication_s = 60.0,
    duty_cycle = 0.02,
))]
#[allow(clippy::too_many_arguments)]
pub fn run_transcranial_fus_planning_from_ritk_ct<'py>(
    py: Python<'py>,
    ct_nifti_path: &str,
    segmentation_nifti_path: Option<&str>,
    element_count: usize,
    frequency_hz: f64,
    radius_m: f64,
    cap_min_polar_rad: f64,
    cap_max_polar_rad: f64,
    brain_sound_speed: f64,
    skull_sound_speed: f64,
    target_peak_pa: f64,
    samples_per_ray: usize,
    skull_hu_threshold: f64,
    body_hu_threshold: f64,
    target_fraction_xyz: Option<(f64, f64, f64)>,
    mechanical_index_bbb: f64,
    sonication_s: f64,
    duty_cycle: f64,
) -> PyResult<Bound<'py, PyDict>> {
    // ── Load CT volume ────────────────────────────────────────────────────────
    let (mut ct, spacing_mm) = load_ritk_nifti(Path::new(ct_nifti_path))?;
    ct.mapv_inplace(|hu| hu.clamp(-1024.0, 3071.0));

    let spacing_m = [
        spacing_mm[0] * 1.0e-3,
        spacing_mm[1] * 1.0e-3,
        spacing_mm[2] * 1.0e-3,
    ];

    // ── Derive tissue masks ───────────────────────────────────────────────────
    let skull_mask = ct.mapv(|hu| hu >= skull_hu_threshold);
    let brain_mask = ct.mapv(|hu| hu >= body_hu_threshold && hu < skull_hu_threshold);

    // ── Load segmentation (tumour mask) if provided ───────────────────────────
    let tumor_mask = if let Some(seg_path) = segmentation_nifti_path {
        let (seg, _) = load_ritk_nifti(Path::new(seg_path))?;
        // Any non-zero label is tumour.
        seg.mapv(|v| v > 0.5)
    } else {
        // Empty tumour mask: planning falls back to focus subspot only.
        ndarray::Array3::from_elem(ct.dim(), false)
    };

    // ── Derive target index from brain centroid ───────────────────────────────
    let target_index = if let Some((x, y, z)) = target_fraction_xyz {
        target_index_from_mask_fraction_3d(&brain_mask, [x, y, z]).map_err(kwavers_to_py)?
    } else {
        brain_centroid(&brain_mask)
    };

    // ── Build config ──────────────────────────────────────────────────────────
    let config = TranscranialFusPlanConfig {
        element_count,
        frequency_hz,
        radius_m,
        cap_min_polar_rad,
        cap_max_polar_rad,
        brain_c: brain_sound_speed,
        skull_c: skull_sound_speed,
        target_peak_pa,
        samples_per_ray,
        mechanical_index_bbb,
        sonication_s,
        duty_cycle,
        ..TranscranialFusPlanConfig::default()
    };

    // ── Run planning (release GIL for heavy compute) ──────────────────────────
    let plan = py
        .detach(|| {
            run_transcranial_fus_planning(
                &ct,
                &skull_mask,
                &brain_mask,
                &tumor_mask,
                spacing_m,
                target_index,
                &config,
            )
        })
        .map_err(kwavers_to_py)?;

    // ── Build output dict ─────────────────────────────────────────────────────
    let out = PyDict::new(py);
    out.set_item("pressure_pa", plan.pressure_pa.into_pyarray(py))?;
    out.set_item("intensity_w_m2", plan.intensity_w_m2.into_pyarray(py))?;
    out.set_item("mechanical_index", plan.mechanical_index.into_pyarray(py))?;
    out.set_item(
        "cavitation_probability",
        plan.cavitation_probability.into_pyarray(py),
    )?;
    out.set_item("phases_rad", plan.phases_rad.into_pyarray(py))?;
    out.set_item("delays_s", plan.delays_s.into_pyarray(py))?;
    out.set_item("skull_lengths_m", plan.skull_lengths_m.into_pyarray(py))?;
    out.set_item("amplitude_weights", plan.amplitude_weights.into_pyarray(py))?;
    out.set_item(
        "element_positions_m",
        plan.element_positions_m.into_pyarray(py),
    )?;
    out.set_item("subspot_indices", plan.subspot_indices.into_pyarray(py))?;
    out.set_item("bbb_dose", plan.bbb_dose.into_pyarray(py))?;
    out.set_item("bbb_permeability", plan.bbb_permeability.into_pyarray(py))?;
    out.set_item(
        "bbb_stable_cavitation",
        plan.bbb_stable_cavitation.into_pyarray(py),
    )?;
    out.set_item("bbb_inertial_risk", plan.bbb_inertial_risk.into_pyarray(py))?;
    out.set_item(
        "focus_index",
        (
            plan.focus_index[0],
            plan.focus_index[1],
            plan.focus_index[2],
        ),
    )?;
    out.set_item("element_count", plan.element_count)?;
    out.set_item("frequency_hz", plan.frequency_hz)?;
    if let Some((x, y, z)) = target_fraction_xyz {
        out.set_item("target_fraction_xyz", (x, y, z))?;
    }
    Ok(out)
}

/// Compute the centroid of the brain mask as integer voxel indices.
/// Falls back to the array centre if the mask is empty.
fn brain_centroid(brain_mask: &ndarray::Array3<bool>) -> [usize; 3] {
    let (nx, ny, nz) = brain_mask.dim();
    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    let mut sz = 0.0_f64;
    let mut n = 0_usize;
    for ((ix, iy, iz), &active) in brain_mask.indexed_iter() {
        if active {
            sx += ix as f64;
            sy += iy as f64;
            sz += iz as f64;
            n += 1;
        }
    }
    if n == 0 {
        return [nx / 2, ny / 2, nz / 2];
    }
    [
        (sx / n as f64).round() as usize,
        (sy / n as f64).round() as usize,
        (sz / n as f64).round() as usize,
    ]
}
