//! Neuromodulation threshold-search bindings.

use kwavers_physics::acoustics::therapy::neuromodulation::{
    BilayerSonophoreQuasistatic, CorticalNeuron, HhParams, Membrane, ThresholdQuery,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Bisection threshold search over a membrane model with the quasi-static source.
fn threshold_for<M: Membrane + Clone>(
    membrane: M,
    v_rest_mv: f64,
    freq_mhz: f64,
    cm0_uf_cm2: f64,
    p_lo_pa: f64,
    p_hi_pa: f64,
    n_iter: usize,
    dt_ms: f64,
    onset_ms: f64,
    offset_ms: f64,
    t_end_ms: f64,
) -> Option<f64> {
    let q = ThresholdQuery {
        membrane,
        v_rest_mv,
        i_bias_ua_cm2: 0.0,
        dt_ms,
        onset_ms,
        offset_ms,
        t_end_ms,
    };
    let make = |p: f64| BilayerSonophoreQuasistatic::new(cm0_uf_cm2, freq_mhz, p, v_rest_mv);
    q.threshold_pressure_pa(p_lo_pa, p_hi_pa, n_iter, &make)
}

/// Minimum acoustic pressure amplitude `Pa` that evokes a post-stimulus action
/// potential, found by bisection with the quasi-static bilayer-sonophore source.
#[pyfunction]
#[pyo3(signature = (
    neuron_class, freq_mhz, p_lo_pa = 50.0e3, p_hi_pa = 800.0e3, n_iter = 8,
    dt_ms = 4.0e-5, onset_ms = 1.0, offset_ms = 8.0, t_end_ms = 16.0, cm0_uf_cm2 = 1.0
))]
#[allow(clippy::too_many_arguments)]
pub fn neuromod_threshold_pressure_pa(
    neuron_class: &str,
    freq_mhz: f64,
    p_lo_pa: f64,
    p_hi_pa: f64,
    n_iter: usize,
    dt_ms: f64,
    onset_ms: f64,
    offset_ms: f64,
    t_end_ms: f64,
    cm0_uf_cm2: f64,
) -> PyResult<Option<f64>> {
    let thr = match neuron_class.to_ascii_lowercase().as_str() {
        "squid" => threshold_for(
            HhParams {
                cm_uf_cm2: cm0_uf_cm2,
                ..HhParams::default()
            },
            -65.0,
            freq_mhz,
            cm0_uf_cm2,
            p_lo_pa,
            p_hi_pa,
            n_iter,
            dt_ms,
            onset_ms,
            offset_ms,
            t_end_ms,
        ),
        "rs" | "regular_spiking" => threshold_for(
            CorticalNeuron::regular_spiking(),
            CorticalNeuron::V_REST_RS_MV,
            freq_mhz,
            cm0_uf_cm2,
            p_lo_pa,
            p_hi_pa,
            n_iter,
            dt_ms,
            onset_ms,
            offset_ms,
            t_end_ms,
        ),
        "fs" | "fast_spiking" => threshold_for(
            CorticalNeuron::fast_spiking(),
            CorticalNeuron::V_REST_FS_MV,
            freq_mhz,
            cm0_uf_cm2,
            p_lo_pa,
            p_hi_pa,
            n_iter,
            dt_ms,
            onset_ms,
            offset_ms,
            t_end_ms,
        ),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown neuron_class '{other}' (expected 'squid', 'rs', or 'fs')"
            )))
        }
    };
    Ok(thr)
}
