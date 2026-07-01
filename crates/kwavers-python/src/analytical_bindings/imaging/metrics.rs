//! IVUS chapter metric bindings.

use kwavers_physics::analytical::imaging;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Chapter 30 IVUS scalar metrics from Rust-owned fields.
#[pyfunction]
#[pyo3(signature = (
    x_m,
    y_m,
    lumen_mask,
    eel_mask,
    plaque_mask,
    bmode_cartesian,
    sound_speed_m_s,
    imaging_frequency_hz,
    therapy_frequency_hz,
    bmode_dynamic_range_db,
    therapy_mechanical_index,
    therapy_peak_delta_t_c,
    therapy_target_to_offtarget_deposition_ratio
))]
#[allow(clippy::too_many_arguments)]
pub fn ivus_chapter_metrics<'py>(
    py: Python<'py>,
    x_m: PyReadonlyArray2<f64>,
    y_m: PyReadonlyArray2<f64>,
    lumen_mask: PyReadonlyArray2<bool>,
    eel_mask: PyReadonlyArray2<bool>,
    plaque_mask: PyReadonlyArray2<bool>,
    bmode_cartesian: PyReadonlyArray2<f64>,
    sound_speed_m_s: f64,
    imaging_frequency_hz: f64,
    therapy_frequency_hz: f64,
    bmode_dynamic_range_db: f64,
    therapy_mechanical_index: f64,
    therapy_peak_delta_t_c: f64,
    therapy_target_to_offtarget_deposition_ratio: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let y = y_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let lumen = lumen_mask
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let eel = eel_mask
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let plaque = plaque_mask
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let bmode = bmode_cartesian
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let metrics = py
        .detach(|| {
            imaging::ivus_chapter_metrics(
                x,
                y,
                lumen,
                eel,
                plaque,
                bmode,
                sound_speed_m_s,
                imaging_frequency_hz,
                therapy_frequency_hz,
                bmode_dynamic_range_db,
                therapy_mechanical_index,
                therapy_peak_delta_t_c,
                therapy_target_to_offtarget_deposition_ratio,
            )
        })
        .map_err(PyValueError::new_err)?;

    let out = PyDict::new(py);
    out.set_item("imaging_wavelength_um", metrics.imaging_wavelength_um)?;
    out.set_item("therapy_wavelength_mm", metrics.therapy_wavelength_mm)?;
    out.set_item("lumen_area_mm2", metrics.lumen_area_mm2)?;
    out.set_item("plaque_area_mm2", metrics.plaque_area_mm2)?;
    out.set_item("bmode_dynamic_range_db", metrics.bmode_dynamic_range_db)?;
    out.set_item(
        "bmode_mean_lumen_intensity",
        metrics.bmode_mean_lumen_intensity,
    )?;
    out.set_item(
        "bmode_mean_wall_intensity",
        metrics.bmode_mean_wall_intensity,
    )?;
    out.set_item("therapy_mechanical_index", metrics.therapy_mechanical_index)?;
    out.set_item("therapy_peak_delta_t_c", metrics.therapy_peak_delta_t_c)?;
    out.set_item(
        "therapy_target_to_offtarget_deposition_ratio",
        metrics.therapy_target_to_offtarget_deposition_ratio,
    )?;
    Ok(out)
}
