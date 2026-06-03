//! PyO3 bindings for `kwavers_physics::analytical::skull`.

use kwavers_physics::analytical::skull as skull_mod;
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute two-way skull insertion loss using a power-law attenuation model.
///
/// Args:
///     f_mhz: Frequency array [MHz].
///     thickness_cm: Skull thickness [cm].
///     alpha0: Attenuation coefficient [dB/(cm·MHz)].
///
/// Returns:
///     Two-way insertion loss array [dB].
#[pyfunction]
#[pyo3(signature = (f_mhz, thickness_cm, alpha0))]
pub fn skull_insertion_loss_two_way_db(
    py: Python<'_>,
    f_mhz: PyReadonlyArray1<f64>,
    thickness_cm: f64,
    alpha0: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let f_s = f_mhz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = skull_mod::skull_insertion_loss_two_way_db(f_s, thickness_cm, alpha0);
    Ok(result.into_pyarray(py).unbind())
}

/// Generate a random phase screen modelling skull aberration.
///
/// Args:
///     n: Number of phase-screen points.
///     sigma_phi_rad: Phase standard deviation [rad].
///     seed: RNG seed for reproducibility.
///
/// Returns:
///     Phase array [rad] of length *n*.
#[pyfunction]
#[pyo3(signature = (n, sigma_phi_rad, seed))]
pub fn skull_phase_screen(
    py: Python<'_>,
    n: usize,
    sigma_phi_rad: f64,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = skull_mod::skull_phase_screen(n, sigma_phi_rad, seed);
    Ok(result.into_pyarray(py).unbind())
}

/// Convert Hounsfield units to sound speed using the Schneider model.
///
/// Args:
///     hu: HU array.
///
/// Returns:
///     Sound speed array [m/s].
#[pyfunction]
#[pyo3(signature = (hu,))]
pub fn hu_to_sound_speed_schneider(
    py: Python<'_>,
    hu: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let h_s = hu
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = skull_mod::hu_to_sound_speed_schneider(h_s);
    Ok(result.into_pyarray(py).unbind())
}

/// Convert Hounsfield units to density using the Schneider model.
///
/// Args:
///     hu: HU array.
///
/// Returns:
///     Density array [kg/m³].
#[pyfunction]
#[pyo3(signature = (hu,))]
pub fn hu_to_density_schneider(
    py: Python<'_>,
    hu: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let h_s = hu
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = skull_mod::hu_to_density_schneider(h_s);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the Strehl ratio for a given wavefront-error standard deviation.
///
/// S ≈ exp(-sigma_phi²)  (Maréchal approximation)
///
/// Args:
///     sigma_phi_rad: RMS wavefront error [rad].
///
/// Returns:
///     Strehl ratio (0–1).
#[pyfunction]
#[pyo3(signature = (sigma_phi_rad,))]
pub fn strehl_ratio(sigma_phi_rad: f64) -> PyResult<f64> {
    Ok(skull_mod::strehl_ratio(sigma_phi_rad))
}

/// Compute skull surface temperature rise due to a heat-flux boundary.
///
/// Args:
///     t_arr: Time array [s].
///     heat_flux: Applied heat flux [W/m²].
///     k_skull: Skull thermal conductivity [W/(m·K)].
///     rho_skull: Skull density [kg/m³].
///     cp_skull: Skull specific heat capacity [J/(kg·K)].
///
/// Returns:
///     Surface temperature-rise array [°C].
#[pyfunction]
#[pyo3(signature = (t_arr, heat_flux, k_skull, rho_skull, cp_skull))]
pub fn skull_surface_temperature_rise(
    py: Python<'_>,
    t_arr: PyReadonlyArray1<f64>,
    heat_flux: f64,
    k_skull: f64,
    rho_skull: f64,
    cp_skull: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result =
        skull_mod::skull_surface_temperature_rise(t_s, heat_flux, k_skull, rho_skull, cp_skull);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the complex transmission coefficient through a skull layer
/// using the transfer-matrix method.
///
/// Args:
///     f_hz: Frequency array [Hz].
///     z_water: Water impedance [Pa·s/m].
///     z_skull: Skull impedance [Pa·s/m].
///     z_brain: Brain impedance [Pa·s/m].
///     c_skull: Skull sound speed [m/s].
///     d_skull_m: Skull thickness [m].
///
/// Returns:
///     Python complex number T = |T| * exp(i*phi).
#[pyfunction]
#[pyo3(signature = (f_hz, z_water, z_skull, z_brain, c_skull, d_skull_m))]
pub fn skull_transfer_matrix_transmission(
    py: Python<'_>,
    f_hz: f64,
    z_water: f64,
    z_skull: f64,
    z_brain: f64,
    c_skull: f64,
    d_skull_m: f64,
) -> PyResult<Py<PyAny>> {
    let c: Complex64 = skull_mod::skull_transfer_matrix_transmission(
        f_hz, z_water, z_skull, z_brain, c_skull, d_skull_m,
    );
    let builtins = py.import("builtins")?;
    let py_complex = builtins.getattr("complex")?;
    Ok(py_complex.call1((c.re, c.im))?.into())
}

/// Compute the skull transmission spectrum (magnitude and phase).
///
/// Args:
///     f_hz: Frequency array [Hz].
///     z_water: Water impedance [Pa·s/m].
///     z_skull: Skull impedance [Pa·s/m].
///     z_brain: Brain impedance [Pa·s/m].
///     c_skull: Skull sound speed [m/s].
///     d_skull_m: Skull thickness [m].
///
/// Returns:
///     (magnitude_array, phase_array_rad) tuple.
#[pyfunction]
#[pyo3(signature = (f_hz, z_water, z_skull, z_brain, c_skull, d_skull_m))]
pub fn skull_transmission_spectrum(
    py: Python<'_>,
    f_hz: PyReadonlyArray1<f64>,
    z_water: f64,
    z_skull: f64,
    z_brain: f64,
    c_skull: f64,
    d_skull_m: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let f_s = f_hz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (mag, phase) =
        skull_mod::skull_transmission_spectrum(f_s, z_water, z_skull, z_brain, c_skull, d_skull_m);
    Ok((
        mag.into_pyarray(py).unbind(),
        phase.into_pyarray(py).unbind(),
    ))
}
