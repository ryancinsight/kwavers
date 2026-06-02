//! PyO3 bindings for `kwavers::physics::analytical::thermal`.

use kwavers::physics::analytical::thermal;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

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

/// Compute the HIFU focal pressure gain (ratio of focal to source pressure).
///
/// Args:
///     aperture_m: Transducer aperture diameter [m].
///     f_number: F-number.
///     freq_hz: Frequency [Hz].
///     c: Sound speed [m/s].
///
/// Returns:
///     Focal pressure gain (dimensionless).
#[pyfunction]
#[pyo3(signature = (aperture_m, f_number, freq_hz, c))]
pub fn hifu_focal_pressure_gain(
    aperture_m: f64,
    f_number: f64,
    freq_hz: f64,
    c: f64,
) -> PyResult<f64> {
    Ok(thermal::hifu_focal_pressure_gain(
        aperture_m, f_number, freq_hz, c,
    ))
}

/// Compute the 2-D Gaussian power-deposition distribution.
///
/// Returns a 2-D array of shape (len(r_arr), len(z_arr)) [W/m³].
///
/// Args:
///     r_arr: Radial positions [m].
///     z_arr: Axial positions [m].
///     freq_hz: Frequency [Hz].
///     z_focus_m: Focal depth [m].
///     p0_pa: Source pressure [Pa].
///     c: Sound speed [m/s].
///     rho: Density [kg/m³].
///     alpha_np_m: Attenuation [Np/m].
///     w0_m: Beam waist at focus [m].
///
/// Returns:
///     Power deposition ndarray [W/m³] of shape (nr, nz).
#[pyfunction]
#[pyo3(signature = (r_arr, z_arr, freq_hz, z_focus_m, p0_pa, c, rho, alpha_np_m, w0_m))]
pub fn gaussian_power_deposition_2d(
    py: Python<'_>,
    r_arr: PyReadonlyArray1<f64>,
    z_arr: PyReadonlyArray1<f64>,
    freq_hz: f64,
    z_focus_m: f64,
    p0_pa: f64,
    c: f64,
    rho: f64,
    alpha_np_m: f64,
    w0_m: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let r_s = r_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let nr = r_s.len();
    let nz = z_s.len();
    let flat = thermal::gaussian_power_deposition_2d(
        r_s, z_s, freq_hz, z_focus_m, p0_pa, c, rho, alpha_np_m, w0_m,
    );
    let arr2d = Array2::from_shape_vec((nr, nz), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr2d.into_pyarray(py).unbind())
}

/// Acoustic intensity depth profile I(z) = I₀·exp(−2·α·z).
///
/// Args:
///     z_arr: Depth positions [m].
///     alpha_np_m: Amplitude attenuation coefficient [Np/m].
///     surface_intensity: Surface intensity I₀ at z=0 [W/m² or normalised].
///
/// Returns:
///     Intensity array [same units as surface_intensity].
#[pyfunction]
#[pyo3(signature = (z_arr, alpha_np_m, surface_intensity))]
pub fn acoustic_intensity_depth_profile(
    py: Python<'_>,
    z_arr: PyReadonlyArray1<f64>,
    alpha_np_m: f64,
    surface_intensity: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = thermal::acoustic_intensity_depth_profile(z_s, alpha_np_m, surface_intensity);
    Ok(result.into_pyarray(py).unbind())
}

/// Volumetric acoustic power deposition Q(z) = 2·α·I₀·exp(−2·α·z).
///
/// Args:
///     z_arr: Depth positions [m].
///     alpha_np_m: Amplitude attenuation coefficient [Np/m].
///     surface_intensity: Surface intensity I₀ at z=0 [W/m²].
///
/// Returns:
///     Power deposition density [W/m³].
#[pyfunction]
#[pyo3(signature = (z_arr, alpha_np_m, surface_intensity))]
pub fn acoustic_power_deposition_depth_profile(
    py: Python<'_>,
    z_arr: PyReadonlyArray1<f64>,
    alpha_np_m: f64,
    surface_intensity: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result =
        thermal::acoustic_power_deposition_depth_profile(z_s, alpha_np_m, surface_intensity);
    Ok(result.into_pyarray(py).unbind())
}

/// Acoustic intensity from peak pressure amplitude: I = p² / (2·ρ·c) [W/m²].
///
/// Computes the Spatial-Peak Pulse-Average Intensity (ISPPA) for a CW plane
/// wave, or ISPTA at unity duty cycle.
///
/// Args:
///     p_field: Peak pressure amplitude field [Pa], any shape, passed as 1-D.
///     rho: Medium density [kg/m³].
///     c: Speed of sound [m/s].
///
/// Returns:
///     Intensity array [W/m²], same length as p_field.
///
/// Reference:
///     Pierce (1989) Acoustics, §1.11.
#[pyfunction]
#[pyo3(signature = (p_field, rho, c))]
pub fn acoustic_intensity_from_amplitude(
    py: Python<'_>,
    p_field: PyReadonlyArray1<f64>,
    rho: f64,
    c: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let p_s = p_field
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = thermal::acoustic_intensity_from_amplitude(p_s, rho, c);
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
    let result =
        thermal::adiabatic_temperature_rise_kelvin(q_s, tau_s, density, specific_heat);
    Ok(result.into_pyarray(py).unbind())
}

/// Convert a flattened acoustic pressure field to volumetric heat-source density.
///
/// Computes Q(x,y,z) = α·p(x,y,z)²/(ρ·c) [W/m³] — the Pennes bioheat
/// source term for a CW or time-averaged pressure field.
///
/// Derivation:  I = p²/(2ρc),  Q = 2α·I  →  Q = α·p²/(ρ·c).
///
/// The input array is accepted and returned in flattened (row-major) order;
/// reshape back to (nx, ny, nz) on the Python side.
///
/// Args:
///     p_field: Pressure amplitude field [Pa], any shape, passed as 1-D.
///     alpha_np_m: Amplitude attenuation coefficient [Np/m].
///     rho: Medium density [kg/m³].
///     c: Speed of sound [m/s].
///
/// Returns:
///     Heat-source density [W/m³], same length as p_field.
///
/// References:
///     Pennes (1948) J. Appl. Physiol. 1, 93.
///     Duck (1990) Physical Properties of Tissue, §5.2.
#[pyfunction]
#[pyo3(signature = (p_field, alpha_np_m, rho, c))]
pub fn acoustic_heat_source_density(
    py: Python<'_>,
    p_field: PyReadonlyArray1<f64>,
    alpha_np_m: f64,
    rho: f64,
    c: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let p_s = p_field
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = thermal::acoustic_heat_source_density(p_s, alpha_np_m, rho, c);
    Ok(result.into_pyarray(py).unbind())
}
