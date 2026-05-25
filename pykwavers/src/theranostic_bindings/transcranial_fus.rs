//! PyO3 bindings for the transcranial FUS planning pipeline.
//!
//! Provides two entry points:
//!
//! - `run_transcranial_fus_planning_from_ritk_ct`: loads CT NIfTI from disk,
//!   derives masks internally, runs the full Rust pipeline.
//! - `run_transcranial_fus_planning_from_arrays`: accepts pre-loaded numpy
//!   arrays (CT HU, skull/brain/tumor masks, spacing, target index), delegates
//!   all physics to Rust.
//! - `transcranial_pennes_thermal_dose_py`: heterogeneous Pennes bioheat
//!   (skull/brain/water tissue properties) from pre-loaded numpy arrays.
//!
//! No wave physics, no Rayleigh integrals, and no finite-difference PDE solvers
//! exist in this file. All computation is delegated to
//! `kwavers::clinical::therapy::theranostic_guidance`.

use kwavers::clinical::therapy::theranostic_guidance::{
    run_transcranial_fus_planning, target_index_from_mask_fraction_3d,
    transcranial_pennes_thermal_dose, TranscranialFusPlanConfig,
};
use numpy::{IntoPyArray, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

use super::helpers::kwavers_to_py;
use crate::ritk_image::load_ritk_nifti;

// ── NIfTI-file entry point ────────────────────────────────────────────────────

/// Run the complete transcranial FUS therapy planning pipeline from a CT NIfTI file.
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
    // Load CT volume.
    let (mut ct, spacing_mm) = load_ritk_nifti(Path::new(ct_nifti_path))?;
    ct.mapv_inplace(|hu| hu.clamp(-1024.0, 3071.0));

    let spacing_m = [
        spacing_mm[0] * 1.0e-3,
        spacing_mm[1] * 1.0e-3,
        spacing_mm[2] * 1.0e-3,
    ];

    // Derive tissue masks.
    let skull_mask = ct.mapv(|hu| hu >= skull_hu_threshold);
    let brain_mask = ct.mapv(|hu| hu >= body_hu_threshold && hu < skull_hu_threshold);

    // Load segmentation (tumour mask) if provided.
    let tumor_mask = if let Some(seg_path) = segmentation_nifti_path {
        let (seg, _) = load_ritk_nifti(Path::new(seg_path))?;
        seg.mapv(|v| v > 0.5)
    } else {
        ndarray::Array3::from_elem(ct.dim(), false)
    };

    // Derive target index from brain centroid.
    let target_index = if let Some((x, y, z)) = target_fraction_xyz {
        target_index_from_mask_fraction_3d(&brain_mask, [x, y, z]).map_err(kwavers_to_py)?
    } else {
        brain_centroid(&brain_mask)
    };

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

    // Release GIL for heavy compute.
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

    plan_to_pydict(py, &plan, target_fraction_xyz)
}

// ── Array entry point ─────────────────────────────────────────────────────────

/// Run the complete transcranial FUS therapy planning pipeline from pre-loaded
/// numpy arrays.
///
/// Accepts CT HU volume, skull/brain/tumor masks, voxel spacing, and target
/// voxel index. Delegates skull ray tracing, phase correction, Rayleigh field
/// synthesis, acoustic observables, GBM subspot raster, and BBB dose to the
/// Rust `run_transcranial_fus_planning` function.
///
/// Returns a dict with all planning fields as numpy arrays, matching the output
/// of `run_transcranial_fus_planning_from_ritk_ct`.
///
/// Parameters
/// ----------
/// ct_hu : ndarray (nx, ny, nz), dtype float64
///     CT Hounsfield unit volume.
/// skull_mask : ndarray (nx, ny, nz), dtype bool
///     Voxels classified as skull cortical bone.
/// brain_mask : ndarray (nx, ny, nz), dtype bool
///     Voxels classified as brain parenchyma.
/// tumor_mask : ndarray (nx, ny, nz), dtype bool
///     Tumour voxels for GBM subspot raster. Pass a zero array for ET VIM.
/// spacing_m : tuple of 3 floats
///     Voxel edge lengths [m].
/// target_index : tuple of 3 ints
///     Target (focus) voxel index (ix, iy, iz).
#[pyfunction]
#[pyo3(signature = (
    ct_hu,
    skull_mask,
    brain_mask,
    tumor_mask,
    spacing_m,
    target_index,
    element_count = 1024,
    frequency_hz = 650_000.0,
    radius_m = 0.150,
    cap_min_polar_rad = 0.22,
    cap_max_polar_rad = 1.18,
    brain_sound_speed = 1540.0,
    skull_sound_speed = 2800.0,
    target_peak_pa = 1_000_000.0,
    samples_per_ray = 192,
    chunk_size = 512,
    mechanical_index_bbb = 0.45,
    sonication_s = 60.0,
    duty_cycle = 0.02,
))]
#[allow(clippy::too_many_arguments)]
pub fn run_transcranial_fus_planning_from_arrays<'py>(
    py: Python<'py>,
    ct_hu: PyReadonlyArray3<'py, f64>,
    skull_mask: PyReadonlyArray3<'py, bool>,
    brain_mask: PyReadonlyArray3<'py, bool>,
    tumor_mask: PyReadonlyArray3<'py, bool>,
    spacing_m: (f64, f64, f64),
    target_index: (usize, usize, usize),
    element_count: usize,
    frequency_hz: f64,
    radius_m: f64,
    cap_min_polar_rad: f64,
    cap_max_polar_rad: f64,
    brain_sound_speed: f64,
    skull_sound_speed: f64,
    target_peak_pa: f64,
    samples_per_ray: usize,
    chunk_size: usize,
    mechanical_index_bbb: f64,
    sonication_s: f64,
    duty_cycle: f64,
) -> PyResult<Bound<'py, PyDict>> {
    // Marshal arrays into owned ndarray storage before releasing GIL.
    let ct = ct_hu.as_array().to_owned();
    let skull = skull_mask.as_array().to_owned();
    let brain = brain_mask.as_array().to_owned();
    let tumor = tumor_mask.as_array().to_owned();
    let spacing = [spacing_m.0, spacing_m.1, spacing_m.2];
    let target = [target_index.0, target_index.1, target_index.2];

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
        chunk_size,
        mechanical_index_bbb,
        sonication_s,
        duty_cycle,
        ..TranscranialFusPlanConfig::default()
    };

    // Release GIL for heavy compute (skull ray tracing + Rayleigh integration).
    let plan = py
        .detach(|| run_transcranial_fus_planning(&ct, &skull, &brain, &tumor, spacing, target, &config))
        .map_err(kwavers_to_py)?;

    plan_to_pydict(py, &plan, None)
}

// ── Thermal entry point ───────────────────────────────────────────────────────

/// Heterogeneous Pennes bioheat thermal dose for transcranial FUS.
///
/// Implements the Pennes bioheat equation with tissue-specific material
/// properties for skull, brain, and background water. Returns peak temperature,
/// final temperature, CEM43 thermal dose, and lesion mask.
///
/// All physics — Laplacian finite differences, perfusion, acoustic heating,
/// CEM43 accumulation — execute in Rust. No Python-side computation.
///
/// Parameters
/// ----------
/// intensity_w_m2 : ndarray (nx, ny, nz), dtype float32
///     Steady-state time-averaged acoustic intensity [W/m²].
/// skull_mask : ndarray (nx, ny, nz), dtype bool
///     Skull cortical bone voxels.
/// brain_mask : ndarray (nx, ny, nz), dtype bool
///     Brain parenchyma voxels.
/// spacing_m : tuple of 3 floats
///     Voxel edge lengths [m].
/// frequency_hz : float
///     Operating frequency [Hz] — used for α→Q conversion.
/// sonication_s : float
///     Sonication duration [s]. Default 12.0.
/// dt_s : float
///     Explicit Euler time step [s]. Default 0.25.
/// baseline_c : float
///     Initial and arterial blood temperature [°C]. Default 37.0.
///
/// Returns
/// -------
/// dict with keys: peak_temperature_c, final_temperature_c, cem43_min, lesion_mask.
#[pyfunction]
#[pyo3(signature = (
    intensity_w_m2,
    skull_mask,
    brain_mask,
    spacing_m,
    frequency_hz,
    sonication_s = 12.0,
    dt_s = 0.25,
    baseline_c = 37.0,
))]
#[allow(clippy::too_many_arguments)]
pub fn transcranial_pennes_thermal_dose_py<'py>(
    py: Python<'py>,
    intensity_w_m2: PyReadonlyArray3<'py, f32>,
    skull_mask: PyReadonlyArray3<'py, bool>,
    brain_mask: PyReadonlyArray3<'py, bool>,
    spacing_m: (f64, f64, f64),
    frequency_hz: f64,
    sonication_s: f64,
    dt_s: f64,
    baseline_c: f64,
) -> PyResult<Bound<'py, PyDict>> {
    // Marshal to owned arrays before releasing GIL.
    let intensity = intensity_w_m2.as_array().to_owned();
    let skull = skull_mask.as_array().to_owned();
    let brain = brain_mask.as_array().to_owned();
    let spacing = [spacing_m.0, spacing_m.1, spacing_m.2];

    // Release GIL for Pennes time-stepping.
    let result = py.detach(|| {
        transcranial_pennes_thermal_dose(
            &intensity,
            &skull,
            &brain,
            spacing,
            frequency_hz,
            sonication_s,
            dt_s,
            baseline_c,
        )
    });

    let out = PyDict::new(py);
    out.set_item("peak_temperature_c", result.peak_temperature_c.into_pyarray(py))?;
    out.set_item("final_temperature_c", result.final_temperature_c.into_pyarray(py))?;
    out.set_item("cem43_min", result.cem43_min.into_pyarray(py))?;
    out.set_item("lesion_mask", result.lesion_mask.into_pyarray(py))?;
    Ok(out)
}

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Serialize a `TranscranialFusPlan` into a Python dict.
fn plan_to_pydict<'py>(
    py: Python<'py>,
    plan: &kwavers::clinical::therapy::theranostic_guidance::TranscranialFusPlan,
    target_fraction_xyz: Option<(f64, f64, f64)>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("pressure_pa", plan.pressure_pa.clone().into_pyarray(py))?;
    out.set_item("intensity_w_m2", plan.intensity_w_m2.clone().into_pyarray(py))?;
    out.set_item("mechanical_index", plan.mechanical_index.clone().into_pyarray(py))?;
    out.set_item(
        "cavitation_probability",
        plan.cavitation_probability.clone().into_pyarray(py),
    )?;
    out.set_item("phases_rad", plan.phases_rad.clone().into_pyarray(py))?;
    out.set_item("delays_s", plan.delays_s.clone().into_pyarray(py))?;
    out.set_item("skull_lengths_m", plan.skull_lengths_m.clone().into_pyarray(py))?;
    out.set_item("amplitude_weights", plan.amplitude_weights.clone().into_pyarray(py))?;
    out.set_item(
        "element_positions_m",
        plan.element_positions_m.clone().into_pyarray(py),
    )?;
    out.set_item("subspot_indices", plan.subspot_indices.clone().into_pyarray(py))?;
    out.set_item("bbb_dose", plan.bbb_dose.clone().into_pyarray(py))?;
    out.set_item("bbb_permeability", plan.bbb_permeability.clone().into_pyarray(py))?;
    out.set_item(
        "bbb_stable_cavitation",
        plan.bbb_stable_cavitation.clone().into_pyarray(py),
    )?;
    out.set_item("bbb_inertial_risk", plan.bbb_inertial_risk.clone().into_pyarray(py))?;
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
