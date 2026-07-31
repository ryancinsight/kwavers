use crate::array_utils::vec_to_pyarray2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use kwavers_signal::SignalWindowType;

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
    let signal = kwavers_signal::tone_burst_series(&kwavers_signal::ToneBurstSpec {
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
    let signals = kwavers_signal::create_cw_signals(t_slice, frequency_hz, amp_slice, phase_slice)
        .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;
    let (rows, cols) = (signals.shape()[0], signals.shape()[1]);
    let data = signals.into_vec();
    vec_to_pyarray2(py, [rows, cols], data)
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
    let window = kwavers_signal::window::get_win(wtype, n, symmetric);
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
    let result = kwavers_signal::add_noise(data, snr_db, seed)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, result).into())
}

pub(super) fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tone_burst, m)?)?;
    m.add_function(wrap_pyfunction!(create_cw_signals, m)?)?;
    m.add_function(wrap_pyfunction!(get_win, m)?)?;
    m.add_function(wrap_pyfunction!(add_noise, m)?)?;
    Ok(())
}
