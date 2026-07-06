//! IVUS therapy field and response bindings.

use kwavers_physics::analytical::imaging;
use numpy::{ToPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// IVUS therapy pressure field for a sector-focused intravascular source.
#[pyfunction]
#[pyo3(signature = (
    radius_m,
    theta_rad,
    catheter_radius_m,
    peak_pressure_pa,
    therapy_azimuth_rad,
    sector_width_rad,
    attenuation_length_m
))]
#[allow(clippy::too_many_arguments)]
pub fn ivus_therapy_pressure_field(
    py: Python<'_>,
    radius_m: PyReadonlyArray1<f64>,
    theta_rad: PyReadonlyArray1<f64>,
    catheter_radius_m: f64,
    peak_pressure_pa: f64,
    therapy_azimuth_rad: f64,
    sector_width_rad: f64,
    attenuation_length_m: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let radius = radius_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let theta = theta_rad
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pressure = py
        .detach(|| {
            imaging::ivus_therapy_pressure_field(
                radius,
                theta,
                catheter_radius_m,
                peak_pressure_pa,
                therapy_azimuth_rad,
                sector_width_rad,
                attenuation_length_m,
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok(pressure.to_pyarray(py).unbind())
}

/// IVUS microbubble delivery fraction from acoustic radiation force.
#[pyfunction]
#[pyo3(signature = (
    range_m,
    attenuation_np_m,
    intensity_w_m2,
    wall_mask,
    target_mask,
    sound_speed_m_s,
    radial_center_m,
    radial_width_m
))]
#[allow(clippy::too_many_arguments)]
pub fn ivus_microbubble_delivery_fraction(
    py: Python<'_>,
    range_m: PyReadonlyArray1<f64>,
    attenuation_np_m: PyReadonlyArray1<f64>,
    intensity_w_m2: PyReadonlyArray1<f64>,
    wall_mask: PyReadonlyArray1<bool>,
    target_mask: PyReadonlyArray1<bool>,
    sound_speed_m_s: f64,
    radial_center_m: f64,
    radial_width_m: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let range = range_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let attenuation = attenuation_np_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let intensity = intensity_w_m2
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let wall = wall_mask
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let target = target_mask
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let delivered = py
        .detach(|| {
            imaging::ivus_microbubble_delivery_fraction(imaging::IvusMicrobubbleDeliveryInput {
                range_m: range,
                attenuation_np_m: attenuation,
                intensity_w_m2: intensity,
                wall_mask: wall,
                target_mask: target,
                sound_speed_m_s,
                radial_center_m,
                radial_width_m,
            })
        })
        .map_err(PyValueError::new_err)?;
    Ok(delivered.to_pyarray(py).unbind())
}

/// IVUS therapy response fields and summary metrics.
#[pyfunction]
#[pyo3(signature = (
    pressure_pa,
    radius_m,
    attenuation_db_cm_mhz,
    eel_mask,
    lumen_mask,
    fibrous_cap_mask,
    lipid_mask,
    plaque_mask,
    catheter_radius_m,
    therapy_frequency_hz,
    therapy_duty_cycle,
    therapy_sonication_s,
    density_kg_m3,
    sound_speed_m_s,
    specific_heat_j_kg_k,
    delivery_radial_center_m,
    delivery_radial_width_m
))]
#[allow(clippy::too_many_arguments)]
pub fn ivus_therapy_response<'py>(
    py: Python<'py>,
    pressure_pa: PyReadonlyArray2<f64>,
    radius_m: PyReadonlyArray2<f64>,
    attenuation_db_cm_mhz: PyReadonlyArray2<f64>,
    eel_mask: PyReadonlyArray2<bool>,
    lumen_mask: PyReadonlyArray2<bool>,
    fibrous_cap_mask: PyReadonlyArray2<bool>,
    lipid_mask: PyReadonlyArray2<bool>,
    plaque_mask: PyReadonlyArray2<bool>,
    catheter_radius_m: f64,
    therapy_frequency_hz: f64,
    therapy_duty_cycle: f64,
    therapy_sonication_s: f64,
    density_kg_m3: f64,
    sound_speed_m_s: f64,
    specific_heat_j_kg_k: f64,
    delivery_radial_center_m: f64,
    delivery_radial_width_m: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let pressure = pressure_pa
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let radius = radius_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let attenuation = attenuation_db_cm_mhz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let eel = eel_mask
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let lumen = lumen_mask
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let cap = fibrous_cap_mask
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let lipid = lipid_mask
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let plaque = plaque_mask
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let response = py
        .detach(|| {
            imaging::ivus_therapy_response(
                pressure,
                radius,
                attenuation,
                eel,
                lumen,
                cap,
                lipid,
                plaque,
                catheter_radius_m,
                therapy_frequency_hz,
                therapy_duty_cycle,
                therapy_sonication_s,
                density_kg_m3,
                sound_speed_m_s,
                specific_heat_j_kg_k,
                delivery_radial_center_m,
                delivery_radial_width_m,
            )
        })
        .map_err(PyValueError::new_err)?;

    let out = PyDict::new(py);
    out.set_item("intensity_w_m2", response.intensity_w_m2.to_pyarray(py))?;
    out.set_item(
        "temperature_rise_k",
        response.temperature_rise_k.to_pyarray(py),
    )?;
    out.set_item("deposition", response.deposition.to_pyarray(py))?;
    out.set_item("mechanical_index", response.mechanical_index)?;
    out.set_item(
        "target_to_offtarget_ratio",
        response.target_to_offtarget_ratio,
    )?;
    out.set_item("peak_delta_t_k", response.peak_delta_t_k)?;
    Ok(out)
}

/// Complete IVUS therapy pressure and response fields for Chapter 30.
#[pyfunction]
#[pyo3(signature = (
    radius_m,
    theta_rad,
    attenuation_db_cm_mhz,
    eel_mask,
    lumen_mask,
    fibrous_cap_mask,
    lipid_mask,
    plaque_mask,
    catheter_radius_m,
    therapy_pressure_pa,
    therapy_azimuth_rad,
    therapy_sector_width_rad,
    pressure_attenuation_length_m,
    therapy_frequency_hz,
    therapy_duty_cycle,
    therapy_sonication_s,
    density_kg_m3,
    sound_speed_m_s,
    specific_heat_j_kg_k,
    delivery_radial_center_m,
    delivery_radial_width_m
))]
#[allow(clippy::too_many_arguments)]
pub fn ivus_therapy_fields<'py>(
    py: Python<'py>,
    radius_m: PyReadonlyArray2<f64>,
    theta_rad: PyReadonlyArray2<f64>,
    attenuation_db_cm_mhz: PyReadonlyArray2<f64>,
    eel_mask: PyReadonlyArray2<bool>,
    lumen_mask: PyReadonlyArray2<bool>,
    fibrous_cap_mask: PyReadonlyArray2<bool>,
    lipid_mask: PyReadonlyArray2<bool>,
    plaque_mask: PyReadonlyArray2<bool>,
    catheter_radius_m: f64,
    therapy_pressure_pa: f64,
    therapy_azimuth_rad: f64,
    therapy_sector_width_rad: f64,
    pressure_attenuation_length_m: f64,
    therapy_frequency_hz: f64,
    therapy_duty_cycle: f64,
    therapy_sonication_s: f64,
    density_kg_m3: f64,
    sound_speed_m_s: f64,
    specific_heat_j_kg_k: f64,
    delivery_radial_center_m: f64,
    delivery_radial_width_m: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let radius = radius_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let theta = theta_rad
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let attenuation = attenuation_db_cm_mhz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let eel = eel_mask
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let lumen = lumen_mask
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let cap = fibrous_cap_mask
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let lipid = lipid_mask
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let plaque = plaque_mask
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let fields = py
        .detach(|| {
            imaging::ivus_therapy_fields(
                radius,
                theta,
                attenuation,
                eel,
                lumen,
                cap,
                lipid,
                plaque,
                catheter_radius_m,
                therapy_pressure_pa,
                therapy_azimuth_rad,
                therapy_sector_width_rad,
                pressure_attenuation_length_m,
                therapy_frequency_hz,
                therapy_duty_cycle,
                therapy_sonication_s,
                density_kg_m3,
                sound_speed_m_s,
                specific_heat_j_kg_k,
                delivery_radial_center_m,
                delivery_radial_width_m,
            )
        })
        .map_err(PyValueError::new_err)?;

    let out = PyDict::new(py);
    out.set_item("pressure_pa", fields.pressure_pa.to_pyarray(py))?;
    out.set_item("intensity_w_m2", fields.intensity_w_m2.to_pyarray(py))?;
    out.set_item(
        "temperature_rise_k",
        fields.temperature_rise_k.to_pyarray(py),
    )?;
    out.set_item("deposition", fields.deposition.to_pyarray(py))?;
    out.set_item("mechanical_index", fields.mechanical_index)?;
    out.set_item(
        "target_to_offtarget_ratio",
        fields.target_to_offtarget_ratio,
    )?;
    out.set_item("peak_delta_t_k", fields.peak_delta_t_k)?;
    Ok(out)
}

