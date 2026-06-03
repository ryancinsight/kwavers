use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use kwavers_domain::signal::SignalWindowType;

#[pyfunction]
#[pyo3(signature = (sample_rate_hz, signal_freq_hz, num_cycles, signal_offset=0, signal_length=None, window="Gaussian", amplitude=1.0, phase=0.0))]
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
        "rectangular" => SignalWindowType::Rectangular,
        "hann" | "hanning" => SignalWindowType::Hann,
        "hamming" => SignalWindowType::Hamming,
        "blackman" => SignalWindowType::Blackman,
        "gaussian" => SignalWindowType::Gaussian,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown window type: {}",
                window
            )))
        }
    };
    let signal =
        kwavers_domain::signal::tone_burst_series(&kwavers_domain::signal::ToneBurstSpec {
            sample_rate_hz,
            signal_freq_hz,
            num_cycles,
            signal_offset,
            signal_length,
            window: window_type,
            amplitude,
            phase,
        })
        .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;
    Ok(PyArray1::from_vec(py, signal).into())
}

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
        kwavers_domain::signal::create_cw_signals(t_slice, frequency_hz, amp_slice, phase_slice)
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;
    Ok(PyArray2::from_owned_array(py, signals).into())
}

#[pyfunction]
#[pyo3(signature = (n, window_type, symmetric=true))]
fn get_win(
    py: Python<'_>,
    n: usize,
    window_type: &str,
    symmetric: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    let wtype = match window_type.to_lowercase().as_str() {
        "rectangular" => SignalWindowType::Rectangular,
        "hann" | "hanning" => SignalWindowType::Hann,
        "hamming" => SignalWindowType::Hamming,
        "blackman" => SignalWindowType::Blackman,
        "gaussian" => SignalWindowType::Gaussian,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown window type: {}",
                window_type
            )))
        }
    };
    let window = kwavers_domain::signal::window::get_win(wtype, n, symmetric);
    Ok(PyArray1::from_vec(py, window).into())
}

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
    let sig_power: f64 = data.iter().map(|&x| x * x).sum::<f64>() / data.len() as f64;
    if sig_power <= 0.0 {
        return Err(PyValueError::new_err("Signal power is zero"));
    }
    let noise_power = sig_power / 10.0f64.powf(snr_db / 10.0);
    let noise_std = noise_power.sqrt();
    let mut state = seed.unwrap_or(42);
    let mut result = Vec::with_capacity(data.len());
    for &s in data.iter() {
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
    Ok(PyArray1::from_vec(py, result).into())
}

pub(super) fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tone_burst, m)?)?;
    m.add_function(wrap_pyfunction!(create_cw_signals, m)?)?;
    m.add_function(wrap_pyfunction!(get_win, m)?)?;
    m.add_function(wrap_pyfunction!(add_noise, m)?)?;
    Ok(())
}
