//! Neuron and NICE response simulation bindings.

use kwavers_physics::acoustics::therapy::neuromodulation::{
    simulate_hh, simulate_nice, simulate_sonic, BilayerSonophore, BilayerSonophoreDynamic,
    BilayerSonophoreQuasistatic, CorticalNeuron, HhParams, NiceConfig, SonicConfig,
};
use numpy::{ToPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Trace4 = (
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
);

/// Simulate a Hodgkin-Huxley neuron under a constant external current.
///
/// Args:
///     i_ext_ua_cm2: External current density [µA/cm²].
///     dt_ms: Integration step [ms].
///     t_end_ms: Simulation duration [ms].
///     v_rest_mv: Resting potential / initial condition [mV] (default −65).
///
/// Returns:
///     (time_ms, voltage_mv, spike_times_ms) as numpy arrays.
#[pyfunction]
#[pyo3(signature = (i_ext_ua_cm2, dt_ms, t_end_ms, v_rest_mv = -65.0))]
pub fn hodgkin_huxley_response(
    py: Python<'_>,
    i_ext_ua_cm2: f64,
    dt_ms: f64,
    t_end_ms: f64,
    v_rest_mv: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    if dt_ms <= 0.0 || t_end_ms <= 0.0 {
        return Err(PyValueError::new_err("dt_ms and t_end_ms must be > 0"));
    }
    let p = HhParams::default();
    let trace = simulate_hh(&p, v_rest_mv, |_| i_ext_ua_cm2, dt_ms, t_end_ms);
    Ok((
        trace.time_ms.to_pyarray(py).unbind(),
        trace.voltage_mv.to_pyarray(py).unbind(),
        trace.spike_times_ms.to_pyarray(py).unbind(),
    ))
}

/// Simulate the NICE pathway with the grounded bilayer-sonophore capacitance
/// source (Plaksin et al. 2014, Eq. 8).
#[pyfunction]
#[pyo3(signature = (
    freq_mhz, deflection_nm, dt_ms, onset_ms, offset_ms, t_end_ms,
    v_rest_mv = -65.0, i_bias_ua_cm2 = 0.0, cm0_uf_cm2 = 1.0
))]
#[allow(clippy::too_many_arguments)]
pub fn nice_bilayer_sonophore_response(
    py: Python<'_>,
    freq_mhz: f64,
    deflection_nm: f64,
    dt_ms: f64,
    onset_ms: f64,
    offset_ms: f64,
    t_end_ms: f64,
    v_rest_mv: f64,
    i_bias_ua_cm2: f64,
    cm0_uf_cm2: f64,
) -> PyResult<Trace4> {
    let cfg = NiceConfig {
        membrane: HhParams {
            cm_uf_cm2: cm0_uf_cm2,
            ..HhParams::default()
        },
        v_rest_mv,
        source: BilayerSonophore::new(cm0_uf_cm2, freq_mhz, deflection_nm * 1.0e-9),
        i_bias_ua_cm2,
        dt_ms,
        onset_ms,
        offset_ms,
        t_end_ms,
    };
    if !cfg.is_valid() {
        return Err(PyValueError::new_err(
            "invalid NICE configuration (check dt>0, t_end>0, onset<=offset, cm0>0)",
        ));
    }
    let trace = simulate_nice(&cfg);
    Ok((
        trace.time_ms.to_pyarray(py).unbind(),
        trace.voltage_mv.to_pyarray(py).unbind(),
        trace.charge_nc_cm2.to_pyarray(py).unbind(),
        trace.spike_times_ms.to_pyarray(py).unbind(),
    ))
}

/// Simulate the NICE pathway with the cycle-averaged SONIC reduction
/// (Lemaire et al. 2019).
#[pyfunction]
#[pyo3(signature = (
    freq_mhz, deflection_nm, dt_ms, onset_ms, offset_ms, t_end_ms,
    v_rest_mv = -65.0, i_bias_ua_cm2 = 0.0, cm0_uf_cm2 = 1.0, cycle_samples = 64
))]
#[allow(clippy::too_many_arguments)]
pub fn nice_sonic_response(
    py: Python<'_>,
    freq_mhz: f64,
    deflection_nm: f64,
    dt_ms: f64,
    onset_ms: f64,
    offset_ms: f64,
    t_end_ms: f64,
    v_rest_mv: f64,
    i_bias_ua_cm2: f64,
    cm0_uf_cm2: f64,
    cycle_samples: usize,
) -> PyResult<Trace4> {
    let cfg = SonicConfig {
        membrane: HhParams {
            cm_uf_cm2: cm0_uf_cm2,
            ..HhParams::default()
        },
        v_rest_mv,
        source: BilayerSonophore::new(cm0_uf_cm2, freq_mhz, deflection_nm * 1.0e-9),
        i_bias_ua_cm2,
        dt_ms,
        cycle_samples,
        onset_ms,
        offset_ms,
        t_end_ms,
    };
    if !cfg.is_valid() {
        return Err(PyValueError::new_err(
            "invalid SONIC configuration (check dt>0, t_end>0, onset<=offset, cm0>0, cycle_samples>=8)",
        ));
    }
    let trace = simulate_sonic(&cfg);
    Ok((
        trace.time_ms.to_pyarray(py).unbind(),
        trace.voltage_mv.to_pyarray(py).unbind(),
        trace.charge_nc_cm2.to_pyarray(py).unbind(),
        trace.spike_times_ms.to_pyarray(py).unbind(),
    ))
}

/// NICE simulation with the pressure-driven quasi-static bilayer sonophore.
#[pyfunction]
#[pyo3(signature = (
    pressure_amp_pa, freq_mhz, dt_ms, onset_ms, offset_ms, t_end_ms,
    v_rest_mv = -65.0, i_bias_ua_cm2 = 0.0, cm0_uf_cm2 = 1.0
))]
#[allow(clippy::too_many_arguments)]
pub fn nice_quasistatic_response(
    py: Python<'_>,
    pressure_amp_pa: f64,
    freq_mhz: f64,
    dt_ms: f64,
    onset_ms: f64,
    offset_ms: f64,
    t_end_ms: f64,
    v_rest_mv: f64,
    i_bias_ua_cm2: f64,
    cm0_uf_cm2: f64,
) -> PyResult<Trace4> {
    let cfg = NiceConfig {
        membrane: HhParams {
            cm_uf_cm2: cm0_uf_cm2,
            ..HhParams::default()
        },
        v_rest_mv,
        source: BilayerSonophoreQuasistatic::new(cm0_uf_cm2, freq_mhz, pressure_amp_pa, v_rest_mv),
        i_bias_ua_cm2,
        dt_ms,
        onset_ms,
        offset_ms,
        t_end_ms,
    };
    if !cfg.is_valid() {
        return Err(PyValueError::new_err(
            "invalid quasi-static NICE configuration",
        ));
    }
    let trace = simulate_nice(&cfg);
    Ok((
        trace.time_ms.to_pyarray(py).unbind(),
        trace.voltage_mv.to_pyarray(py).unbind(),
        trace.charge_nc_cm2.to_pyarray(py).unbind(),
        trace.spike_times_ms.to_pyarray(py).unbind(),
    ))
}

/// NICE simulation with the exact transient bilayer sonophore.
#[pyfunction]
#[pyo3(signature = (
    pressure_amp_pa, freq_mhz, dt_ms, onset_ms, offset_ms, t_end_ms,
    v_rest_mv = -65.0, i_bias_ua_cm2 = 0.0, cm0_uf_cm2 = 1.0
))]
#[allow(clippy::too_many_arguments)]
pub fn nice_dynamic_response(
    py: Python<'_>,
    pressure_amp_pa: f64,
    freq_mhz: f64,
    dt_ms: f64,
    onset_ms: f64,
    offset_ms: f64,
    t_end_ms: f64,
    v_rest_mv: f64,
    i_bias_ua_cm2: f64,
    cm0_uf_cm2: f64,
) -> PyResult<Trace4> {
    let cfg = NiceConfig {
        membrane: HhParams {
            cm_uf_cm2: cm0_uf_cm2,
            ..HhParams::default()
        },
        v_rest_mv,
        source: BilayerSonophoreDynamic::new(cm0_uf_cm2, freq_mhz, pressure_amp_pa, v_rest_mv),
        i_bias_ua_cm2,
        dt_ms,
        onset_ms,
        offset_ms,
        t_end_ms,
    };
    if !cfg.is_valid() {
        return Err(PyValueError::new_err("invalid dynamic NICE configuration"));
    }
    let trace = simulate_nice(&cfg);
    Ok((
        trace.time_ms.to_pyarray(py).unbind(),
        trace.voltage_mv.to_pyarray(py).unbind(),
        trace.charge_nc_cm2.to_pyarray(py).unbind(),
        trace.spike_times_ms.to_pyarray(py).unbind(),
    ))
}

/// SONIC simulation on a Pospischil cortical neuron.
#[pyfunction]
#[pyo3(signature = (
    neuron_class, freq_mhz, deflection_nm, dt_ms, onset_ms, offset_ms, t_end_ms,
    i_bias_ua_cm2 = 0.0, cycle_samples = 64
))]
#[allow(clippy::too_many_arguments)]
pub fn cortical_sonic_response(
    py: Python<'_>,
    neuron_class: &str,
    freq_mhz: f64,
    deflection_nm: f64,
    dt_ms: f64,
    onset_ms: f64,
    offset_ms: f64,
    t_end_ms: f64,
    i_bias_ua_cm2: f64,
    cycle_samples: usize,
) -> PyResult<Trace4> {
    let (neuron, v_rest) = match neuron_class.to_ascii_lowercase().as_str() {
        "rs" | "regular_spiking" => (
            CorticalNeuron::regular_spiking(),
            CorticalNeuron::V_REST_RS_MV,
        ),
        "fs" | "fast_spiking" => (CorticalNeuron::fast_spiking(), CorticalNeuron::V_REST_FS_MV),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown neuron_class '{other}' (expected 'rs' or 'fs')"
            )))
        }
    };
    let cfg = SonicConfig {
        membrane: neuron,
        v_rest_mv: v_rest,
        source: BilayerSonophore::new(neuron.cm0_uf_cm2, freq_mhz, deflection_nm * 1.0e-9),
        i_bias_ua_cm2,
        dt_ms,
        cycle_samples,
        onset_ms,
        offset_ms,
        t_end_ms,
    };
    if !cfg.is_valid() {
        return Err(PyValueError::new_err(
            "invalid cortical SONIC configuration (check dt>0, t_end>0, onset<=offset, cycle_samples>=8)",
        ));
    }
    let trace = simulate_sonic(&cfg);
    Ok((
        trace.time_ms.to_pyarray(py).unbind(),
        trace.voltage_mv.to_pyarray(py).unbind(),
        trace.charge_nc_cm2.to_pyarray(py).unbind(),
        trace.spike_times_ms.to_pyarray(py).unbind(),
    ))
}

