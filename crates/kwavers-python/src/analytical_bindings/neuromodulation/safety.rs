//! Neuromodulation exposure safety and dosimetry bindings.

use kwavers_physics::acoustics::therapy::neuromodulation::{itrusst_assess, PulseTrainProtocol};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// ITRUSST biophysical-safety assessment of a transcranial-US exposure.
#[pyfunction]
#[pyo3(signature = (mechanical_index, peak_temp_rise_c, cem43_brain_min))]
pub fn itrusst_safety(
    py: Python<'_>,
    mechanical_index: f64,
    peak_temp_rise_c: f64,
    cem43_brain_min: f64,
) -> PyResult<Py<PyDict>> {
    let a = itrusst_assess(mechanical_index, peak_temp_rise_c, cem43_brain_min);
    let dict = PyDict::new(py);
    dict.set_item("mechanical_ok", a.mechanical_ok)?;
    dict.set_item("thermal_ok", a.thermal_ok)?;
    dict.set_item("overall_ok", a.overall_ok)?;
    Ok(dict.unbind())
}

/// Pulse-train dosimetry for an ultrasonic-neuromodulation protocol.
#[pyfunction]
#[pyo3(signature = (
    carrier_freq_hz, pulse_length_s, prf_hz, burst_duration_s, burst_interval_s,
    num_bursts, peak_pressure_pa, density_kg_m3, sound_speed_m_s
))]
#[allow(clippy::too_many_arguments)]
pub fn pulse_train_dosimetry(
    py: Python<'_>,
    carrier_freq_hz: f64,
    pulse_length_s: f64,
    prf_hz: f64,
    burst_duration_s: f64,
    burst_interval_s: f64,
    num_bursts: u32,
    peak_pressure_pa: f64,
    density_kg_m3: f64,
    sound_speed_m_s: f64,
) -> PyResult<Py<PyDict>> {
    let protocol = PulseTrainProtocol {
        carrier_freq_hz,
        pulse_length_s,
        pulse_repetition_freq_hz: prf_hz,
        burst_duration_s,
        burst_interval_s,
        num_bursts,
    };
    if !protocol.is_valid() {
        return Err(PyValueError::new_err(
            "invalid pulse-train protocol (check f,PL,PRF>0; PL<=1/PRF; BD>0; BI>=0; N>=1)",
        ));
    }
    let d = protocol.dosimetry(peak_pressure_pa, density_kg_m3, sound_speed_m_s);
    let dict = PyDict::new(py);
    dict.set_item("isppa_w_cm2", d.isppa_w_cm2)?;
    dict.set_item("ispba_w_cm2", d.ispba_w_cm2)?;
    dict.set_item("ispta_w_cm2", d.ispta_w_cm2)?;
    dict.set_item("mechanical_index", d.mechanical_index)?;
    dict.set_item("total_duty_cycle", d.total_duty_cycle)?;
    dict.set_item("total_time_s", d.total_time_s)?;
    dict.set_item("within_fda_limits", d.within_fda_limits())?;
    Ok(dict.unbind())
}
