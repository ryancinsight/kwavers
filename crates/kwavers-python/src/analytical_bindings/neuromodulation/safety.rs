//! Neuromodulation exposure safety and dosimetry bindings.

use aequitas::systems::si::quantities::{
    Frequency, MassDensity, Pressure, TemperatureDifference, Time, Velocity,
};
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
    let a = itrusst_assess(
        mechanical_index,
        TemperatureDifference::from_base(peak_temp_rise_c),
        cem43_brain_min,
    );
    let dict = PyDict::new(py);
    dict.set_item("mechanical_ok", a.mechanical_ok)?;
    dict.set_item("thermal_ok", a.thermal_ok)?;
    dict.set_item("overall_ok", a.overall_ok)?;
    Ok(dict.unbind())
}

/// Pulse-train dosimetry for an ultrasonic-neuromodulation protocol.
#[pyfunction]
#[pyo3(signature = (
    carrier_frequency, pulse_length, prf_hz, burst_duration, burst_interval,
    num_bursts, peak_pressure_pa, density_kg_m3, sound_speed_m_s
))]
#[allow(clippy::too_many_arguments)]
pub fn pulse_train_dosimetry(
    py: Python<'_>,
    carrier_frequency: f64,
    pulse_length: f64,
    prf_hz: f64,
    burst_duration: f64,
    burst_interval: f64,
    num_bursts: u32,
    peak_pressure_pa: f64,
    density_kg_m3: f64,
    sound_speed_m_s: f64,
) -> PyResult<Py<PyDict>> {
    let protocol = PulseTrainProtocol {
        carrier_frequency: Frequency::from_base(carrier_frequency),
        pulse_length: Time::from_base(pulse_length),
        pulse_repetition_frequency: Frequency::from_base(prf_hz),
        burst_duration: Time::from_base(burst_duration),
        burst_interval: Time::from_base(burst_interval),
        num_bursts,
    };
    if !protocol.is_valid() {
        return Err(PyValueError::new_err(
            "invalid pulse-train protocol (check f,PL,PRF>0; PL<=1/PRF; BD>0; BI>=0; N>=1)",
        ));
    }
    let d = protocol.dosimetry(
        Pressure::from_base(peak_pressure_pa),
        MassDensity::from_base(density_kg_m3),
        Velocity::from_base(sound_speed_m_s),
    );
    let dict = PyDict::new(py);
    dict.set_item("isppa", d.isppa.into_base())?;
    dict.set_item("ispba", d.ispba.into_base())?;
    dict.set_item("ispta", d.ispta.into_base())?;
    dict.set_item("mechanical_index", d.mechanical_index)?;
    dict.set_item("total_duty_cycle", d.total_duty_cycle)?;
    dict.set_item("total_time", d.total_time.into_base())?;
    dict.set_item("within_fda_limits", d.within_fda_limits())?;
    Ok(dict.unbind())
}
