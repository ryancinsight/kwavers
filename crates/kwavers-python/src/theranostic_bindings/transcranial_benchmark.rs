//! PyO3 binding for the skull-adaptive transcranial benchmark.

use crate::breast_fwi_bindings::complex_compat::{leto2_to_nd2, leto3_to_nd3};
use kwavers_diagnostics::reconstruction::transcranial_ust::{
    resample_head_volume, select_head_slice,
};
use kwavers_therapy::therapy::theranostic_guidance::{
    run_skull_adaptive_transcranial_benchmark, target_index_from_mask_fraction_3d,
    SkullAdaptiveBenchmarkConfig, TranscranialFusPlanConfig,
};
use leto::Array3;
use numpy::ToPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

use super::helpers::kwavers_to_py;
use crate::ritk_image::load_ritk_nifti;

#[pyfunction]
#[pyo3(signature = (
    ct_nifti_path,
    grid_size = 32,
    slice_index = None,
    element_count = 1024,
    frequency_hz = 650_000.0,
    radius_m = 0.150,
    aperture_diameter_m = 0.120,
    target_peak_pa = 1_000_000.0,
    samples_per_ray = 192,
    chunk_size = 512,
    skull_hu_threshold = 300.0,
    body_hu_threshold = -350.0,
    target_fraction_xyz = None,
    minimum_active_elements = 8,
    cap_min_polar_rad = 0.22,
    cap_max_polar_rad = 1.18,
    brain_sound_speed = 1540.0,
    skull_sound_speed = 2800.0
))]
#[allow(clippy::too_many_arguments)]
pub fn run_transcranial_skull_adaptive_benchmark_from_ritk_ct<'py>(
    py: Python<'py>,
    ct_nifti_path: &str,
    grid_size: usize,
    slice_index: Option<usize>,
    element_count: usize,
    frequency_hz: f64,
    radius_m: f64,
    aperture_diameter_m: f64,
    target_peak_pa: f64,
    samples_per_ray: usize,
    chunk_size: usize,
    skull_hu_threshold: f64,
    body_hu_threshold: f64,
    target_fraction_xyz: Option<(f64, f64, f64)>,
    minimum_active_elements: usize,
    cap_min_polar_rad: f64,
    cap_max_polar_rad: f64,
    brain_sound_speed: f64,
    skull_sound_speed: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let (mut ct, spacing_mm) = load_ritk_nifti(Path::new(ct_nifti_path))?;
    for hu in ct.iter_mut() {
        *hu = hu.clamp(-1024.0, 3071.0);
    }
    let selected_slice = match slice_index {
        Some(index) => index,
        None => select_head_slice(&ct).map_err(kwavers_to_py)?,
    };
    let resampled =
        resample_head_volume(&ct, spacing_mm, selected_slice, grid_size).map_err(kwavers_to_py)?;
    let ct_hu = resampled.hu;
    let spacing_m = [resampled.spacing_m; 3];
    let skull_mask = ct_hu.mapv(|hu| hu >= skull_hu_threshold);
    let brain_mask = ct_hu.mapv(|hu| hu >= body_hu_threshold && hu < skull_hu_threshold);
    let target_index = if let Some((x, y, z)) = target_fraction_xyz {
        target_index_from_mask_fraction_3d(&brain_mask, [x, y, z]).map_err(kwavers_to_py)?
    } else {
        brain_centroid(&brain_mask)
    };
    let config = SkullAdaptiveBenchmarkConfig {
        fus: TranscranialFusPlanConfig {
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
            ..TranscranialFusPlanConfig::default()
        },
        aperture_diameter_m,
        minimum_active_elements,
    };

    let result = py
        .detach(|| {
            run_skull_adaptive_transcranial_benchmark(
                &ct_hu,
                &skull_mask,
                &brain_mask,
                spacing_m,
                target_index,
                &config,
            )
        })
        .map_err(kwavers_to_py)?;
    let metrics_result = result.metrics;
    let placement_result = result.placement;

    let out = PyDict::new(py);
    out.set_item(
        "reference_pressure_pa",
        leto3_to_nd3(result.reference_pressure_pa).to_pyarray(py),
    )?;
    out.set_item(
        "baseline_pressure_pa",
        leto3_to_nd3(result.baseline_pressure_pa).to_pyarray(py),
    )?;
    out.set_item(
        "phases_rad",
        numpy::ndarray::Array1::try_from(result.phases_rad)
            .expect("invariant: contiguous phases")
            .to_pyarray(py),
    )?;
    out.set_item(
        "delays_s",
        numpy::ndarray::Array1::try_from(result.delays_s)
            .expect("invariant: contiguous delays")
            .to_pyarray(py),
    )?;
    out.set_item(
        "skull_lengths_m",
        numpy::ndarray::Array1::try_from(result.skull_lengths_m)
            .expect("invariant: contiguous skull lengths")
            .to_pyarray(py),
    )?;
    out.set_item(
        "amplitude_weights",
        numpy::ndarray::Array1::try_from(result.amplitude_weights)
            .expect("invariant: contiguous amplitude weights")
            .to_pyarray(py),
    )?;
    out.set_item(
        "element_positions_m",
        leto2_to_nd2(placement_result.element_positions_m).to_pyarray(py),
    )?;
    out.set_item(
        "active_elements",
        numpy::ndarray::Array1::try_from(placement_result.active_elements)
            .expect("invariant: contiguous active elements")
            .to_pyarray(py),
    )?;
    out.set_item(
        "focus_index",
        (
            result.focus_index[0],
            result.focus_index[1],
            result.focus_index[2],
        ),
    )?;
    out.set_item(
        "spacing_m",
        (
            result.spacing_m[0],
            result.spacing_m[1],
            result.spacing_m[2],
        ),
    )?;
    out.set_item("frequency_hz", result.frequency_hz)?;
    out.set_item("target_peak_pa", result.target_peak_pa)?;
    out.set_item("cap_min_polar_rad", cap_min_polar_rad)?;
    out.set_item("cap_max_polar_rad", cap_max_polar_rad)?;
    out.set_item("brain_sound_speed", brain_sound_speed)?;
    out.set_item("skull_sound_speed", skull_sound_speed)?;
    if let Some((x, y, z)) = target_fraction_xyz {
        out.set_item("target_fraction_xyz", (x, y, z))?;
    }
    out.set_item("source_slice_index", selected_slice)?;
    out.set_item("source_volume_index", resampled.source_volume_index)?;
    out.set_item(
        "benchmark_model",
        "ct_conditioned_skull_aware_aperture_vs_uncorrected_baseline",
    )?;

    let metrics = PyDict::new(py);
    metrics.set_item("relative_l2", metrics_result.relative_l2)?;
    metrics.set_item(
        "focal_position_error_m",
        metrics_result.focal_position_error_m,
    )?;
    metrics.set_item(
        "max_pressure_error_percent",
        metrics_result.max_pressure_error_percent,
    )?;
    metrics.set_item("reference_peak_pa", metrics_result.reference_peak_pa)?;
    metrics.set_item("candidate_peak_pa", metrics_result.candidate_peak_pa)?;
    metrics.set_item(
        "reference_focus_index",
        (
            metrics_result.reference_focus_index[0],
            metrics_result.reference_focus_index[1],
            metrics_result.reference_focus_index[2],
        ),
    )?;
    metrics.set_item(
        "candidate_focus_index",
        (
            metrics_result.candidate_focus_index[0],
            metrics_result.candidate_focus_index[1],
            metrics_result.candidate_focus_index[2],
        ),
    )?;
    out.set_item("metrics", metrics)?;

    let placement = PyDict::new(py);
    placement.set_item(
        "aperture_anchor_index",
        placement_result.aperture_anchor_index,
    )?;
    placement.set_item(
        "active_element_count",
        placement_result.active_element_count,
    )?;
    placement.set_item("aperture_diameter_m", placement_result.aperture_diameter_m)?;
    placement.set_item(
        "radius_of_curvature_m",
        placement_result.radius_of_curvature_m,
    )?;
    placement.set_item("focal_length_m", placement_result.focal_length_m)?;
    placement.set_item("mean_skull_length_m", placement_result.mean_skull_length_m)?;
    placement.set_item(
        "mean_amplitude_weight",
        placement_result.mean_amplitude_weight,
    )?;
    placement.set_item(
        "min_amplitude_weight",
        placement_result.min_amplitude_weight,
    )?;
    placement.set_item(
        "max_amplitude_weight",
        placement_result.max_amplitude_weight,
    )?;
    out.set_item("placement", placement)?;

    let comparison = PyDict::new(py);
    comparison.set_item(
        "reference_setup",
        "TFUScapes stores pseudo-CT, transducer coordinates, and k-Wave pressure maps",
    )?;
    comparison.set_item("kwavers_setup", "kwavers stores CT HU, active focused-bowl aperture coordinates, skull-aware pressure, and uncorrected baseline")?;
    comparison.set_item(
        "not_implemented",
        "DeepTFUS neural surrogate training and TFUScapes-scale dataset generation",
    )?;
    out.set_item("paper_structural_comparison", comparison)?;
    Ok(out)
}

fn brain_centroid(brain_mask: &Array3<bool>) -> [usize; 3] {
    let [nx, ny, nz] = brain_mask.shape();
    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    let mut sz = 0.0_f64;
    let mut n = 0_usize;
    for ([ix, iy, iz], active) in brain_mask.indexed_iter() {
        if *active {
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
