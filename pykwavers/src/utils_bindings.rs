// ============================================================================
// Utility Functions (Exposed to Python)
// ============================================================================

use crate::Grid;
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Generate a tone burst signal.
///
/// Returns a numpy array containing the tone burst waveform.
#[pyfunction]
#[pyo3(signature = (sample_rate_hz, signal_freq_hz, num_cycles, signal_offset=0, signal_length=None, window="Hanning", amplitude=1.0, phase=0.0))]
#[allow(clippy::too_many_arguments)]
fn tone_burst(
    py: Python<'_>,
    sample_rate_hz: f64,
    signal_freq_hz: f64,
    num_cycles: f64,
    signal_offset: usize,
    signal_length: Option<usize>,
    window: &str,
    amplitude: f64,
    phase: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let window_type = match window.to_lowercase().as_str() {
        "rectangular" => kwavers::domain::signal::WindowType::Rectangular,
        "hann" | "hanning" => kwavers::domain::signal::WindowType::Hann,
        "hamming" => kwavers::domain::signal::WindowType::Hamming,
        "blackman" => kwavers::domain::signal::WindowType::Blackman,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown window type: {}",
                window
            )))
        }
    };

    let signal = kwavers::domain::signal::tone_burst_series(
        sample_rate_hz,
        signal_freq_hz,
        num_cycles,
        signal_offset,
        signal_length,
        window_type,
        amplitude,
        phase,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;

    let array = PyArray1::from_vec(py, signal);
    Ok(array.into())
}

/// Create continuous wave signals with optional phase shifts
#[pyfunction]
fn create_cw_signals(
    py: Python<'_>,
    t: PyReadonlyArray1<f64>,
    frequency_hz: f64,
    amplitudes: PyReadonlyArray1<f64>,
    phases: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let t_slice = t
        .as_slice()
        .map_err(|_| PyValueError::new_err("Failed to read time array"))?;
    let amp_slice = amplitudes
        .as_slice()
        .map_err(|_| PyValueError::new_err("Failed to read amplitudes"))?;
    let phase_slice = phases
        .as_slice()
        .map_err(|_| PyValueError::new_err("Failed to read phases"))?;

    let signals =
        kwavers::domain::signal::create_cw_signals(t_slice, frequency_hz, amp_slice, phase_slice)
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;

    let array = PyArray2::from_owned_array(py, signals);
    Ok(array.into())
}

/// Get a window function.
///
/// Returns a numpy array containing the window coefficients.
#[pyfunction]
#[pyo3(signature = (n, window_type, symmetric=true))]
fn get_win(
    py: Python<'_>,
    n: usize,
    window_type: &str,
    symmetric: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    let wtype = match window_type.to_lowercase().as_str() {
        "rectangular" => kwavers::domain::signal::WindowType::Rectangular,
        "hann" | "hanning" => kwavers::domain::signal::WindowType::Hann,
        "hamming" => kwavers::domain::signal::WindowType::Hamming,
        "blackman" => kwavers::domain::signal::WindowType::Blackman,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown window type: {}",
                window_type
            )))
        }
    };

    let window = kwavers::domain::signal::window::get_win(wtype, n, symmetric);
    let array = PyArray1::from_vec(py, window);
    Ok(array.into())
}

// ============================================================================
// Geometry Functions
// ============================================================================

/// Create a 2D circular disc mask (filled circle)
#[pyfunction]
fn make_disc(
    py: Python<'_>,
    grid: &Grid,
    center: (f64, f64, f64),
    radius: f64,
) -> PyResult<Py<PyArray3<bool>>> {
    let center_arr = [center.0, center.1, center.2];
    let mask = kwavers::math::geometry::make_disc(&grid.inner, center_arr, radius)
        .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;

    let array = PyArray3::from_owned_array(py, mask);
    Ok(array.into())
}

/// Create a 3D spherical ball mask
#[pyfunction]
fn make_ball(
    py: Python<'_>,
    grid: &Grid,
    center: (f64, f64, f64),
    radius: f64,
) -> PyResult<Py<PyArray3<bool>>> {
    let center_arr = [center.0, center.1, center.2];
    let mask = kwavers::math::geometry::make_ball(&grid.inner, center_arr, radius)
        .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;

    let array = PyArray3::from_owned_array(py, mask);
    Ok(array.into())
}

/// Create a 3D spherical mask (alias for make_ball, MATLAB: makeSphere)
#[pyfunction]
fn make_sphere(
    py: Python<'_>,
    grid: &Grid,
    center: (f64, f64, f64),
    radius: f64,
) -> PyResult<Py<PyArray3<bool>>> {
    make_ball(py, grid, center, radius)
}

/// Create a 2D circle outline (shell) mask
///
/// Unlike make_disc which creates a filled circle, make_circle creates only
/// the circle perimeter (outline). Matches k-Wave's makeCircle.
///
/// Args:
///     grid: Grid defining spatial discretization
///     center: Center point (x, y, z) in meters
///     radius: Circle radius in meters
///     thickness: Shell thickness in grid points (default: 1)
#[pyfunction]
#[pyo3(signature = (grid, center, radius, thickness=1))]
fn make_circle(
    py: Python<'_>,
    grid: &Grid,
    center: (f64, f64, f64),
    radius: f64,
    thickness: usize,
) -> PyResult<Py<PyArray3<bool>>> {
    let center_arr = [center.0, center.1, center.2];
    let mask = kwavers::math::geometry::make_circle(&grid.inner, center_arr, radius, thickness)
        .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;

    let array = PyArray3::from_owned_array(py, mask);
    Ok(array.into())
}

/// Create a line mask connecting two points
#[pyfunction]
fn make_line(
    py: Python<'_>,
    grid: &Grid,
    start: (f64, f64, f64),
    end: (f64, f64, f64),
) -> PyResult<Py<PyArray3<bool>>> {
    let start_arr = [start.0, start.1, start.2];
    let end_arr = [end.0, end.1, end.2];
    let mask = kwavers::math::geometry::make_line(&grid.inner, start_arr, end_arr)
        .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;

    let array = PyArray3::from_owned_array(py, mask);
    Ok(array.into())
}

// ============================================================================
// Unit Conversion Functions
// ============================================================================

/// Convert acoustic attenuation from dB/(MHz^y cm) to Nepers/((rad/s)^y m)
/// Matches k-Wave API behavior exactly.
#[pyfunction]
#[pyo3(signature = (db, y=1.0))]
fn db2neper(db: f64, y: f64) -> f64 {
    let neper_per_db = 10.0f64.ln() / 20.0;
    db * (100.0 * neper_per_db) / (2.0 * std::f64::consts::PI * 1e6).powf(y)
}

/// Convert acoustic attenuation from Nepers/((rad/s)^y m) to dB/(MHz^y cm)
/// Matches k-Wave API behavior exactly.
#[pyfunction]
#[pyo3(signature = (neper, y=1.0))]
fn neper2db(neper: f64, y: f64) -> f64 {
    let db_per_neper = 20.0 / 10.0f64.ln();
    neper * (db_per_neper / 100.0) * (2.0 * std::f64::consts::PI * 1e6).powf(y)
}

/// Convert frequency (Hz) to wavenumber (rad/m)
///
/// k = 2π f / c
///
/// Args:
///     frequency: Frequency in Hz
///     sound_speed: Sound speed in m/s
///
/// Returns:
///     Wavenumber in rad/m
#[pyfunction]
fn freq2wavenumber(frequency: f64, sound_speed: f64) -> PyResult<f64> {
    if sound_speed <= 0.0 {
        return Err(PyValueError::new_err("Sound speed must be positive"));
    }
    if frequency < 0.0 {
        return Err(PyValueError::new_err("Frequency must be non-negative"));
    }
    Ok(2.0 * std::f64::consts::PI * frequency / sound_speed)
}

/// Convert Hounsfield units to density (kg/m³)
///
/// Piecewise linear fit to experimental CT data (k-wave compatible).
///
/// Args:
///     hu: Hounsfield unit value (raw CT number)
///
/// Returns:
///     Density in kg/m³
#[pyfunction]
fn hounsfield2density(hu: f64) -> f64 {
    kwavers::core::constants::hounsfield::HounsfieldUnits::to_density(hu)
}

/// Convert Hounsfield units to sound speed (m/s)
///
/// Uses Mast (2000): c = (density(HU) + 349) / 0.893
///
/// Args:
///     hu: Hounsfield unit value (raw CT number)
///
/// Returns:
///     Sound speed in m/s
#[pyfunction]
fn hounsfield2soundspeed(hu: f64) -> f64 {
    kwavers::core::constants::hounsfield::HounsfieldUnits::to_sound_speed(hu)
}

// ============================================================================
// Water Property Functions (Temperature-Dependent)
// ============================================================================

/// Compute water sound speed at a given temperature.
///
/// Compute water sound speed at a given temperature.
///
/// Uses Marczak (1997) 5th-order polynomial.
/// Valid range: 0–95 °C.
///
/// Args:
///     temp_celsius: Temperature in degrees Celsius
///
/// Returns:
///     Sound speed in m/s
#[pyfunction]
fn water_sound_speed(temp_celsius: f64) -> f64 {
    kwavers::core::constants::water::WaterProperties::sound_speed(temp_celsius)
}

/// Compute water density at a given temperature.
///
/// Uses Jones & Harris (1992) 4th-order polynomial for air-saturated water.
/// Valid range: 5–40 °C.
///
/// Args:
///     temp_celsius: Temperature in degrees Celsius
///
/// Returns:
///     Density in kg/m³
#[pyfunction]
fn water_density(temp_celsius: f64) -> f64 {
    kwavers::core::constants::water::WaterProperties::density(temp_celsius)
}

/// Compute water absorption at a given frequency and temperature.
///
/// Uses Pinkerton (1949) model: 7th-order polynomial in temperature,
/// quadratic in frequency.  Input is Hz; output is Np/m.
///
/// Args:
///     frequency: Frequency in Hz
///     temp_celsius: Temperature in degrees Celsius (0–60 °C)
///
/// Returns:
///     Absorption in Np/m
#[pyfunction]
fn water_absorption(frequency: f64, temp_celsius: f64) -> f64 {
    let freq_mhz = frequency / 1e6;
    // Pinkerton model gives dB/cm.  Convert to Np/m:
    //   Np/m = (dB/cm) / 8.686 * 100
    let db_per_cm = kwavers::core::constants::water::WaterProperties::absorption_pinkerton(
        freq_mhz,
        temp_celsius,
    );
    db_per_cm / 8.686 * 100.0
}

/// Compute water nonlinear parameter B/A at a given temperature.
///
/// Uses Beyer (1960) 4th-order polynomial fit.
/// Valid range: 0–100 °C.
///
/// Args:
///     temp_celsius: Temperature in degrees Celsius
///
/// Returns:
///     B/A dimensionless nonlinearity parameter
#[pyfunction]
fn water_nonlinearity(temp_celsius: f64) -> f64 {
    kwavers::core::constants::water::WaterProperties::nonlinear_parameter(temp_celsius)
}

/// Add Gaussian noise to a signal at a specified SNR level.
///
/// Generates additive white Gaussian noise (AWGN) scaled to achieve the
/// requested signal-to-noise ratio relative to the input signal power.
///
/// Args:
///     signal: Input signal as a numpy array
///     snr_db: Desired signal-to-noise ratio in decibels
///     seed: Optional random seed for reproducibility
///
/// Returns:
///     Noisy signal as a numpy array
#[pyfunction]
#[pyo3(signature = (signal, snr_db, seed=None))]
fn add_noise(
    py: Python<'_>,
    signal: PyReadonlyArray1<f64>,
    snr_db: f64,
    seed: Option<u64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let data = signal
        .as_slice()
        .map_err(|_| PyValueError::new_err("Failed to read signal array"))?;

    // Compute signal power
    let sig_power: f64 = data.iter().map(|&x| x * x).sum::<f64>() / data.len() as f64;

    if sig_power <= 0.0 {
        return Err(PyValueError::new_err("Signal power is zero"));
    }

    // Compute noise power from SNR: SNR = 10*log10(Psig/Pnoise)
    let noise_power = sig_power / 10.0f64.powf(snr_db / 10.0);
    let noise_std = noise_power.sqrt();

    // Generate noise using simple xorshift PRNG (deterministic if seeded)
    let mut state = seed.unwrap_or(42);
    let mut result = Vec::with_capacity(data.len());

    for &s in data.iter() {
        // Box-Muller transform for Gaussian samples
        // Generate two uniform samples
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u1 = (state as f64) / (u64::MAX as f64);

        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u2 = (state as f64) / (u64::MAX as f64);

        let u1_clamped = u1.max(1e-300);
        let z = (-2.0 * u1_clamped.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        result.push(s + noise_std * z);
    }

    let array = PyArray1::from_vec(py, result);
    Ok(array.into())
}

/// Register all utility functions with the Python module
pub fn register_utils(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    // Signal generation
    m.add_function(wrap_pyfunction!(tone_burst, m)?)?;
    m.add_function(wrap_pyfunction!(create_cw_signals, m)?)?;
    m.add_function(wrap_pyfunction!(get_win, m)?)?;
    // Geometry
    m.add_function(wrap_pyfunction!(make_disc, m)?)?;
    m.add_function(wrap_pyfunction!(make_ball, m)?)?;
    m.add_function(wrap_pyfunction!(make_sphere, m)?)?;
    m.add_function(wrap_pyfunction!(make_circle, m)?)?;
    m.add_function(wrap_pyfunction!(make_line, m)?)?;
    // Unit conversion
    m.add_function(wrap_pyfunction!(db2neper, m)?)?;
    m.add_function(wrap_pyfunction!(neper2db, m)?)?;
    m.add_function(wrap_pyfunction!(freq2wavenumber, m)?)?;
    m.add_function(wrap_pyfunction!(hounsfield2density, m)?)?;
    m.add_function(wrap_pyfunction!(hounsfield2soundspeed, m)?)?;
    // Water properties
    m.add_function(wrap_pyfunction!(water_sound_speed, m)?)?;
    m.add_function(wrap_pyfunction!(water_density, m)?)?;
    m.add_function(wrap_pyfunction!(water_absorption, m)?)?;
    m.add_function(wrap_pyfunction!(water_nonlinearity, m)?)?;
    // Signal processing
    m.add_function(wrap_pyfunction!(add_noise, m)?)?;
    Ok(())
}
