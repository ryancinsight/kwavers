//! IVUS B-mode RF, scan-conversion, and display bindings.

use kwavers_physics::analytical::imaging;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Polar IVUS RF fixture from phantom backscatter and attenuation fields.
#[pyfunction]
#[pyo3(signature = (
    x_m,
    y_m,
    backscatter,
    attenuation_db_cm_mhz,
    r_axis_m,
    theta_axis_rad,
    catheter_radius_m,
    frequency_hz,
    ring_amplitude=0.10,
    ring_width_m=0.22e-3
))]
#[allow(clippy::too_many_arguments)]
pub fn ivus_polar_bmode_rf(
    py: Python<'_>,
    x_m: PyReadonlyArray2<f64>,
    y_m: PyReadonlyArray2<f64>,
    backscatter: PyReadonlyArray2<f64>,
    attenuation_db_cm_mhz: PyReadonlyArray2<f64>,
    r_axis_m: PyReadonlyArray1<f64>,
    theta_axis_rad: PyReadonlyArray1<f64>,
    catheter_radius_m: f64,
    frequency_hz: f64,
    ring_amplitude: f64,
    ring_width_m: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let x = x_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let y = y_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let back = backscatter
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let attenuation = attenuation_db_cm_mhz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let radius = r_axis_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let theta = theta_axis_rad
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let rf = py
        .detach(|| {
            imaging::ivus_polar_bmode_rf(
                x,
                y,
                back,
                attenuation,
                radius,
                theta,
                catheter_radius_m,
                frequency_hz,
                ring_amplitude,
                ring_width_m,
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok(rf.into_pyarray(py).unbind())
}

/// Nearest-neighbour IVUS polar-to-Cartesian scan conversion.
#[pyfunction]
#[pyo3(signature = (polar, r_axis_m, theta_axis_rad, radius_m, theta_rad))]
pub fn ivus_scan_convert(
    py: Python<'_>,
    polar: PyReadonlyArray2<f64>,
    r_axis_m: PyReadonlyArray1<f64>,
    theta_axis_rad: PyReadonlyArray1<f64>,
    radius_m: PyReadonlyArray2<f64>,
    theta_rad: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let polar_values = polar
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let radius_axis = r_axis_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let theta_axis = theta_axis_rad
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let radius = radius_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let theta = theta_rad
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let image = py
        .detach(|| imaging::ivus_scan_convert(polar_values, radius_axis, theta_axis, radius, theta))
        .map_err(PyValueError::new_err)?;
    Ok(image.into_pyarray(py).unbind())
}

/// Complete IVUS B-mode RF-to-display fixture for Chapter 30.
#[pyfunction]
#[pyo3(signature = (
    x_m,
    y_m,
    backscatter,
    attenuation_db_cm_mhz,
    r_axis_m,
    theta_axis_rad,
    radius_m,
    theta_m_rad,
    catheter_radius_m,
    frequency_hz,
    floor_db=-60.0,
    ring_amplitude=0.10,
    ring_width_m=0.22e-3
))]
#[allow(clippy::too_many_arguments)]
pub fn ivus_bmode_image<'py>(
    py: Python<'py>,
    x_m: PyReadonlyArray2<f64>,
    y_m: PyReadonlyArray2<f64>,
    backscatter: PyReadonlyArray2<f64>,
    attenuation_db_cm_mhz: PyReadonlyArray2<f64>,
    r_axis_m: PyReadonlyArray1<f64>,
    theta_axis_rad: PyReadonlyArray1<f64>,
    radius_m: PyReadonlyArray2<f64>,
    theta_m_rad: PyReadonlyArray2<f64>,
    catheter_radius_m: f64,
    frequency_hz: f64,
    floor_db: f64,
    ring_amplitude: f64,
    ring_width_m: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let y = y_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let back = backscatter
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let attenuation = attenuation_db_cm_mhz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let radius_axis = r_axis_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let theta_axis = theta_axis_rad
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let radius = radius_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let theta = theta_m_rad
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let image = py
        .detach(|| {
            imaging::ivus_bmode_image(
                x,
                y,
                back,
                attenuation,
                radius_axis,
                theta_axis,
                radius,
                theta,
                catheter_radius_m,
                frequency_hz,
                floor_db,
                ring_amplitude,
                ring_width_m,
            )
        })
        .map_err(PyValueError::new_err)?;

    let out = PyDict::new(py);
    out.set_item("rf", image.rf.into_pyarray(py))?;
    out.set_item("envelope", image.envelope.into_pyarray(py))?;
    out.set_item("db", image.db.into_pyarray(py))?;
    out.set_item("polar", image.polar.into_pyarray(py))?;
    out.set_item("cartesian", image.cartesian.into_pyarray(py))?;
    Ok(out)
}
