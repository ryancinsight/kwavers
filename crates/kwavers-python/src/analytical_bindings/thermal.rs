//! PyO3 bindings for `kwavers_physics::analytical::thermal`.

mod acoustic;

use kwavers_physics::analytical::thermal;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pub use acoustic::{
    acoustic_heat_source_density, acoustic_intensity_depth_profile,
    acoustic_intensity_from_amplitude, acoustic_power_deposition_depth_profile,
    acoustic_pressure_amplitude_from_intensity, gaussian_power_deposition_2d,
    hifu_focal_pressure_gain,
};

/// Simulate focal temperature rise using the Pennes bioheat model.
///
/// Args:
///     t_arr: Time array [s].
///     acoustic_power_w: Absorbed acoustic power [W].
///     focal_volume_m3: Focal volume [m³].
///     k_tissue: Tissue thermal conductivity [W/(m·K)].
///     rho_tissue: Tissue density [kg/m³].
///     cp_tissue: Tissue specific heat [J/(kg·K)].
///     wb_perfusion: Blood perfusion rate [kg/(m³·s)].
///     rho_blood: Blood density [kg/m³].
///     cb_blood: Blood specific heat [J/(kg·K)].
///     t_body_c: Body temperature [°C].
///
/// Returns:
///     Temperature array [°C].
#[pyfunction]
#[pyo3(signature = (t_arr, acoustic_power_w, focal_volume_m3, k_tissue, rho_tissue, cp_tissue, wb_perfusion, rho_blood, cb_blood, t_body_c))]
pub fn bioheat_focal_temperature_rise(
    py: Python<'_>,
    t_arr: PyReadonlyArray1<f64>,
    acoustic_power_w: f64,
    focal_volume_m3: f64,
    k_tissue: f64,
    rho_tissue: f64,
    cp_tissue: f64,
    wb_perfusion: f64,
    rho_blood: f64,
    cb_blood: f64,
    t_body_c: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = thermal::bioheat_focal_temperature_rise(
        t_s,
        acoustic_power_w,
        focal_volume_m3,
        k_tissue,
        rho_tissue,
        cp_tissue,
        wb_perfusion,
        rho_blood,
        cb_blood,
        t_body_c,
    );
    Ok(result.into_pyarray(py).unbind())
}

/// One-dimensional steady-state Pennes slab temperature profile.
///
/// Args:
///     x_arr: Depth positions [m].
///     slab_thickness_m: Slab thickness [m].
///     thermal_conductivity: Tissue conductivity [W/(m*K)].
///     blood_perfusion: Blood perfusion rate [1/s].
///     rho_blood: Blood density [kg/m3].
///     cp_blood: Blood specific heat [J/(kg*K)].
///     body_temperature_c: Boundary/body temperature [deg C].
///     heat_source_w_m3: Uniform volumetric heating [W/m3].
///
/// Returns:
///     Temperature profile [deg C].
#[pyfunction]
#[pyo3(signature = (
    x_arr,
    slab_thickness_m,
    thermal_conductivity,
    blood_perfusion,
    rho_blood,
    cp_blood,
    body_temperature_c,
    heat_source_w_m3
))]
pub fn pennes_steady_state_temperature_profile(
    py: Python<'_>,
    x_arr: PyReadonlyArray1<f64>,
    slab_thickness_m: f64,
    thermal_conductivity: f64,
    blood_perfusion: f64,
    rho_blood: f64,
    cp_blood: f64,
    body_temperature_c: f64,
    heat_source_w_m3: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_s = x_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if !slab_thickness_m.is_finite() || slab_thickness_m <= 0.0 {
        return Err(PyValueError::new_err(
            "slab_thickness_m must be finite and positive",
        ));
    }
    if !thermal_conductivity.is_finite() || thermal_conductivity <= 0.0 {
        return Err(PyValueError::new_err(
            "thermal_conductivity must be finite and positive",
        ));
    }
    if !blood_perfusion.is_finite() || blood_perfusion < 0.0 {
        return Err(PyValueError::new_err(
            "blood_perfusion must be finite and non-negative",
        ));
    }
    if !rho_blood.is_finite() || rho_blood <= 0.0 {
        return Err(PyValueError::new_err(
            "rho_blood must be finite and positive",
        ));
    }
    if !cp_blood.is_finite() || cp_blood <= 0.0 {
        return Err(PyValueError::new_err(
            "cp_blood must be finite and positive",
        ));
    }
    if !body_temperature_c.is_finite() {
        return Err(PyValueError::new_err("body_temperature_c must be finite"));
    }
    if !heat_source_w_m3.is_finite() {
        return Err(PyValueError::new_err("heat_source_w_m3 must be finite"));
    }
    if x_s
        .iter()
        .any(|&x| !x.is_finite() || x < 0.0 || x > slab_thickness_m)
    {
        return Err(PyValueError::new_err(
            "x_arr entries must be finite depths inside [0, slab_thickness_m]",
        ));
    }
    let result = thermal::pennes_steady_state_temperature_profile(
        x_s,
        slab_thickness_m,
        thermal_conductivity,
        blood_perfusion,
        rho_blood,
        cp_blood,
        body_temperature_c,
        heat_source_w_m3,
    );
    Ok(result.into_pyarray(py).unbind())
}

/// Adiabatic (no-perfusion) single-pulse temperature rise from a heat source.
///
/// Short-pulse limit of the Pennes bioheat equation where conduction and
/// perfusion are negligible:
///
///     dT_i = Q_i * tau_i / (density * specific_heat)   [K]
///
/// Args:
///     q_arr: Heat-source density array [W/m3].
///     tau_arr: Pulse duration array [s], same length as q_arr.
///     density: Tissue density [kg/m3].
///     specific_heat: Tissue specific heat [J/(kg*K)].
///
/// Returns:
///     Temperature rise array [K], same length as q_arr.
///
/// Reference:
///     Pennes (1948) J. Appl. Physiol. 1, 93 (no-perfusion limit).
#[pyfunction]
#[pyo3(signature = (q_arr, tau_arr, density, specific_heat))]
pub fn adiabatic_temperature_rise_kelvin(
    py: Python<'_>,
    q_arr: PyReadonlyArray1<f64>,
    tau_arr: PyReadonlyArray1<f64>,
    density: f64,
    specific_heat: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let q_s = q_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let tau_s = tau_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = thermal::adiabatic_temperature_rise_kelvin(q_s, tau_s, density, specific_heat);
    Ok(result.into_pyarray(py).unbind())
}
