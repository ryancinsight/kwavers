//! PyO3 bindings for `kwavers::physics::analytical::photoacoustics`.

use kwavers::physics::analytical::photoacoustics;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Return the molar absorption spectrum of oxyhaemoglobin (HbO2).
///
/// Args:
///     wavelength_nm: Wavelength array [nm].
///
/// Returns:
///     Molar absorption coefficient array [cm⁻¹/M].
#[pyfunction]
#[pyo3(signature = (wavelength_nm,))]
pub fn hbo2_molar_absorption(
    py: Python<'_>,
    wavelength_nm: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let w_s = wavelength_nm
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = photoacoustics::hbo2_molar_absorption(w_s);
    Ok(result.into_pyarray(py).unbind())
}

/// Return the molar absorption spectrum of deoxyhaemoglobin (Hb).
///
/// Args:
///     wavelength_nm: Wavelength array [nm].
///
/// Returns:
///     Molar absorption coefficient array [cm⁻¹/M].
#[pyfunction]
#[pyo3(signature = (wavelength_nm,))]
pub fn hb_molar_absorption(
    py: Python<'_>,
    wavelength_nm: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let w_s = wavelength_nm
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = photoacoustics::hb_molar_absorption(w_s);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the Grüneisen parameter of water as a function of temperature.
///
/// Args:
///     t_celsius: Temperature array [°C].
///
/// Returns:
///     Grüneisen parameter array (dimensionless).
#[pyfunction]
#[pyo3(signature = (t_celsius,))]
pub fn gruneisen_parameter_water(
    py: Python<'_>,
    t_celsius: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_celsius
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = photoacoustics::gruneisen_parameter_water(t_s);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the photoacoustic pressure signal from an absorbing sphere.
///
/// Args:
///     t_arr: Time array [s].
///     r0_m: Sphere radius [m].
///     gamma: Grüneisen parameter.
///     mua_per_m: Absorption coefficient [1/m].
///     c: Sound speed [m/s].
///     r_det_m: Detector distance [m].
///     initial_pressure_pa: Initial pressure rise [Pa].
///
/// Returns:
///     Pressure signal array [Pa].
#[pyfunction]
#[pyo3(signature = (t_arr, r0_m, gamma, mua_per_m, c, r_det_m, initial_pressure_pa))]
pub fn pa_sphere_pressure_signal(
    py: Python<'_>,
    t_arr: PyReadonlyArray1<f64>,
    r0_m: f64,
    gamma: f64,
    mua_per_m: f64,
    c: f64,
    r_det_m: f64,
    initial_pressure_pa: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = photoacoustics::pa_sphere_pressure_signal(
        t_s,
        r0_m,
        gamma,
        mua_per_m,
        c,
        r_det_m,
        initial_pressure_pa,
    );
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the photoacoustic axial resolution.
///
/// δz ≈ 0.88 * c / bandwidth
///
/// Args:
///     bandwidth_hz: Transducer bandwidth [Hz].
///     c: Sound speed [m/s].
///
/// Returns:
///     Axial resolution [m].
#[pyfunction]
#[pyo3(signature = (bandwidth_hz, c))]
pub fn pa_axial_resolution(bandwidth_hz: f64, c: f64) -> PyResult<f64> {
    Ok(photoacoustics::pa_axial_resolution(bandwidth_hz, c))
}

/// Estimate chromophore concentrations by least-squares spectral unmixing.
///
/// Solves: spectra_matrix @ concentrations ≈ measurements
///
/// Args:
///     spectra_matrix: Absorption spectra matrix (n_wavelengths × n_chromophores).
///     measurements: Measured PA signals (length n_wavelengths).
///
/// Returns:
///     Concentration vector (length n_chromophores).
#[pyfunction]
#[pyo3(signature = (spectra_matrix, measurements))]
pub fn spectroscopic_unmixing_lstsq(
    py: Python<'_>,
    spectra_matrix: PyReadonlyArray2<f64>,
    measurements: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let sm = spectra_matrix.as_array();
    let (nrows, _ncols) = sm.dim();
    let spectra_vecs: Vec<Vec<f64>> = (0..nrows).map(|i| sm.row(i).to_vec()).collect();
    let meas_slice = measurements
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = photoacoustics::spectroscopic_unmixing_lstsq(&spectra_vecs, meas_slice);
    Ok(result.into_pyarray(py).unbind())
}
