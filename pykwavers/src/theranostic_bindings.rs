//! Python bindings for same-device therapy/imaging FWI simulations.

use kwavers::solver::inverse::seismic::brain_helmet::{resample_head_slice, select_head_slice};
use kwavers::solver::inverse::seismic::theranostic::{
    placement_metrics, prepare_abdominal_slice, prepare_brain_slice, run_theranostic_fwi,
    AnatomyKind, DevicePlacementMetrics, ReconstructionMetrics, TheranosticFwiConfig,
};
use ndarray::{Array1, Array3};
use numpy::IntoPyArray;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::wrap_pyfunction;
use std::path::Path;

use crate::ritk_image::load_ritk_nifti;

#[pyfunction]
#[pyo3(signature = (
    ct_nifti_path,
    segmentation_nifti_path = None,
    anatomy = "brain",
    grid_size = 64,
    element_count = None,
    iterations = 12,
    frequencies_hz = None,
    receiver_offsets = None,
    source_pressure_pa = None,
    noise_fraction = 0.012
))]
fn run_theranostic_fwi_from_ritk<'py>(
    py: Python<'py>,
    ct_nifti_path: &str,
    segmentation_nifti_path: Option<&str>,
    anatomy: &str,
    grid_size: usize,
    element_count: Option<usize>,
    iterations: usize,
    frequencies_hz: Option<Vec<f64>>,
    receiver_offsets: Option<Vec<usize>>,
    source_pressure_pa: Option<f64>,
    noise_fraction: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let anatomy = AnatomyKind::from_name(anatomy).map_err(kwavers_to_py)?;
    let (mut ct, spacing_mm) = load_ritk_nifti(Path::new(ct_nifti_path))?;
    ct.mapv_inplace(|hu| hu.clamp(-1024.0, 3071.0));
    let mut config = TheranosticFwiConfig::new(anatomy);
    config.grid_size = grid_size;
    config.iterations = iterations;
    config.noise_fraction = noise_fraction;
    if let Some(count) = element_count {
        config.element_count = count;
    }
    if let Some(freqs) = frequencies_hz {
        config.frequencies_hz = freqs;
    }
    if let Some(offsets) = receiver_offsets {
        config.receiver_offsets = offsets;
    }
    if let Some(pressure) = source_pressure_pa {
        config.source_pressure_pa = pressure;
    }

    let prepared = match anatomy {
        AnatomyKind::Brain => {
            let selected = select_head_slice(&ct).map_err(kwavers_to_py)?;
            let resampled =
                resample_head_slice(&ct, spacing_mm, selected, grid_size).map_err(kwavers_to_py)?;
            prepare_brain_slice(resampled.hu, resampled.spacing_m, selected)
                .map_err(kwavers_to_py)?
        }
        AnatomyKind::Liver | AnatomyKind::Kidney => {
            let seg_path = segmentation_nifti_path.ok_or_else(|| {
                PyValueError::new_err("segmentation_nifti_path is required for liver and kidney")
            })?;
            let (seg, _) = load_ritk_nifti(Path::new(seg_path))?;
            let labels = labels_from_volume(seg);
            prepare_abdominal_slice(anatomy, &ct, &labels, spacing_mm, grid_size)
                .map_err(kwavers_to_py)?
        }
    };
    let result = py
        .detach(|| run_theranostic_fwi(prepared, &config))
        .map_err(kwavers_to_py)?;
    result_to_dict(py, result, &config)
}

fn labels_from_volume(volume: Array3<f64>) -> Array3<i16> {
    volume.mapv(|value| value.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16)
}

fn result_to_dict<'py>(
    py: Python<'py>,
    result: kwavers::solver::inverse::seismic::theranostic::TheranosticFwiResult,
    config: &TheranosticFwiConfig,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let prepared = result.prepared;
    let layout = result.layout;
    let placement = placement_metrics(&layout, &prepared.body_mask, prepared.spacing_m);
    out.set_item("anatomy", prepared.anatomy.label())?;
    out.set_item("device_model", layout.model_name.clone())?;
    out.set_item("ct_hu", prepared.ct_hu.into_pyarray(py))?;
    out.set_item("label", prepared.label.into_pyarray(py))?;
    out.set_item("sound_speed_m_s", prepared.sound_speed_m_s.into_pyarray(py))?;
    out.set_item(
        "attenuation_np_per_m_mhz",
        prepared.attenuation_np_per_m_mhz.into_pyarray(py),
    )?;
    out.set_item("body_mask", prepared.body_mask.into_pyarray(py))?;
    out.set_item("organ_mask", prepared.organ_mask.into_pyarray(py))?;
    out.set_item("target_mask", prepared.target_mask.into_pyarray(py))?;
    out.set_item("exposure", result.exposure.into_pyarray(py))?;
    out.set_item("lesion_target", result.lesion_target.into_pyarray(py))?;
    out.set_item(
        "anatomy_reconstruction",
        result.anatomy_reconstruction.into_pyarray(py),
    )?;
    out.set_item(
        "active_lesion_reconstruction",
        result.active_lesion_reconstruction.into_pyarray(py),
    )?;
    out.set_item(
        "subharmonic_reconstruction",
        result.subharmonic_reconstruction.into_pyarray(py),
    )?;
    out.set_item(
        "harmonic_reconstruction",
        result.harmonic_reconstruction.into_pyarray(py),
    )?;
    out.set_item(
        "ultraharmonic_reconstruction",
        result.ultraharmonic_reconstruction.into_pyarray(py),
    )?;
    out.set_item(
        "fused_reconstruction",
        result.fused_reconstruction.into_pyarray(py),
    )?;
    out.set_item(
        "therapy_x_m",
        point_axis(&layout.therapy_elements, true).into_pyarray(py),
    )?;
    out.set_item(
        "therapy_y_m",
        point_axis(&layout.therapy_elements, false).into_pyarray(py),
    )?;
    out.set_item(
        "imaging_x_m",
        point_axis(&layout.imaging_receivers, true).into_pyarray(py),
    )?;
    out.set_item(
        "imaging_y_m",
        point_axis(&layout.imaging_receivers, false).into_pyarray(py),
    )?;
    out.set_item("focus_m", (layout.focus_m.x_m, layout.focus_m.y_m))?;
    out.set_item(
        "skin_contact_m",
        (layout.skin_contact_m.x_m, layout.skin_contact_m.y_m),
    )?;
    out.set_item("spacing_m", prepared.spacing_m)?;
    out.set_item("source_slice_index", prepared.source_slice_index)?;
    out.set_item("element_count", config.element_count)?;
    out.set_item("frequencies_hz", config.frequencies_hz.clone())?;
    out.set_item("receiver_offsets", config.receiver_offsets.clone())?;
    out.set_item("source_pressure_pa", config.source_pressure_pa)?;
    out.set_item("geometry_model", layout.model_name.clone())?;
    out.set_item("placement_metrics", placement_dict(py, &placement)?)?;
    out.set_item("operator_model", "finite_frequency_same_aperture_fwi_rtm")?;
    out.set_item("measurements", result.measurements)?;
    out.set_item("active_voxels", result.active_voxels)?;
    out.set_item(
        "objective_history",
        Array1::from(result.objective_history).into_pyarray(py),
    )?;
    let metrics = PyDict::new(py);
    metrics.set_item("anatomy", metric_dict(py, &result.anatomy_metrics)?)?;
    metrics.set_item("active_lesion", metric_dict(py, &result.active_metrics)?)?;
    metrics.set_item("subharmonic", metric_dict(py, &result.subharmonic_metrics)?)?;
    metrics.set_item("harmonic", metric_dict(py, &result.harmonic_metrics)?)?;
    metrics.set_item(
        "ultraharmonic",
        metric_dict(py, &result.ultraharmonic_metrics)?,
    )?;
    metrics.set_item("fusion", metric_dict(py, &result.fused_metrics)?)?;
    out.set_item("metrics", metrics)?;
    Ok(out)
}

fn placement_dict<'py>(
    py: Python<'py>,
    metrics: &DevicePlacementMetrics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("min_body_clearance_m", metrics.min_body_clearance_m)?;
    out.set_item("mean_body_clearance_m", metrics.mean_body_clearance_m)?;
    out.set_item("max_body_clearance_m", metrics.max_body_clearance_m)?;
    out.set_item(
        "skin_contact_to_nearest_aperture_m",
        metrics.skin_contact_to_nearest_aperture_m,
    )?;
    Ok(out)
}

fn point_axis(
    points: &[kwavers::solver::inverse::seismic::theranostic::Point2],
    x_axis: bool,
) -> Array1<f64> {
    Array1::from(
        points
            .iter()
            .map(|point| if x_axis { point.x_m } else { point.y_m })
            .collect::<Vec<_>>(),
    )
}

fn metric_dict<'py>(
    py: Python<'py>,
    metrics: &ReconstructionMetrics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("pearson", metrics.pearson)?;
    out.set_item("nrmse", metrics.nrmse)?;
    out.set_item("dice_equal_area", metrics.dice_equal_area)?;
    out.set_item("cnr", metrics.cnr)?;
    Ok(out)
}

fn kwavers_to_py(err: kwavers::core::error::KwaversError) -> PyErr {
    PyRuntimeError::new_err(format!("kwavers theranostic FWI failed: {err}"))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_theranostic_fwi_from_ritk, m)?)?;
    Ok(())
}
