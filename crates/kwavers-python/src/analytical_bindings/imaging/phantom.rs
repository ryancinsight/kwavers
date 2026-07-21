//! IVUS phantom construction bindings.

use crate::array_utils::vec_to_pyarray2;
use kwavers_physics::analytical::imaging;
use numpy::Element;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Generate the deterministic IVUS vessel phantom used by Chapter 30.
///
/// Returns a dictionary of `(n, n)` arrays. Python book code maps the arrays
/// into its plotting dataclass and performs no anatomy, material, or speckle
/// generation.
#[pyfunction]
#[pyo3(signature = (n=384, fov_m=12.0e-3, catheter_radius_m=0.55e-3, therapy_azimuth_rad=-0.72, seed=30))]
pub fn ivus_vessel_phantom<'py>(
    py: Python<'py>,
    n: usize,
    fov_m: f64,
    catheter_radius_m: f64,
    therapy_azimuth_rad: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let phantom =
        imaging::ivus_vessel_phantom(n, fov_m, catheter_radius_m, therapy_azimuth_rad, seed)
            .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item("x_m", array2(py, n, phantom.x_m)?)?;
    out.set_item("y_m", array2(py, n, phantom.y_m)?)?;
    out.set_item("radius_m", array2(py, n, phantom.radius_m)?)?;
    out.set_item("theta_rad", array2(py, n, phantom.theta_rad)?)?;
    out.set_item("labels", array2(py, n, phantom.labels)?)?;
    out.set_item(
        "sound_speed_m_s",
        array2(py, n, phantom.sound_speed_m_s)?,
    )?;
    out.set_item(
        "density_kg_m3",
        array2(py, n, phantom.density_kg_m3)?,
    )?;
    out.set_item(
        "attenuation_db_cm_mhz",
        array2(py, n, phantom.attenuation_db_cm_mhz)?,
    )?;
    out.set_item(
        "backscatter",
        array2(py, n, phantom.backscatter)?,
    )?;
    out.set_item("lumen_mask", array2(py, n, phantom.lumen_mask)?)?;
    out.set_item("eel_mask", array2(py, n, phantom.eel_mask)?)?;
    out.set_item(
        "plaque_mask",
        array2(py, n, phantom.plaque_mask)?,
    )?;
    out.set_item(
        "fibrous_cap_mask",
        array2(py, n, phantom.fibrous_cap_mask)?,
    )?;
    out.set_item("lipid_mask", array2(py, n, phantom.lipid_mask)?)?;
    out.set_item(
        "calcium_mask",
        array2(py, n, phantom.calcium_mask)?,
    )?;
    Ok(out)
}

fn array2<T: Element + Copy>(
    py: Python<'_>,
    n: usize,
    values: Vec<T>,
) -> PyResult<Py<numpy::PyArray2<T>>> {
    vec_to_pyarray2(py, [n, n], values)
}
