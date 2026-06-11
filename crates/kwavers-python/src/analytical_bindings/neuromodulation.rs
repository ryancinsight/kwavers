//! PyO3 bindings for `kwavers_physics::acoustics::therapy::neuromodulation`
//! (Hodgkin–Huxley neuron, NICE intramembrane-cavitation coupling, bilayer
//! sonophore, and pulse-train dosimetry).

use kwavers_physics::acoustics::therapy::neuromodulation::{
    bls_capacitance, itrusst_assess, quasistatic_deflection, rest_gap, simulate_hh, simulate_nice,
    simulate_sonic, BilayerSonophore, BilayerSonophoreDynamic, BilayerSonophoreQuasistatic,
    CorticalNeuron, HhParams, Membrane, NiceConfig, PulseTrainProtocol, SonicConfig,
    ThresholdQuery,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Simulate a Hodgkin–Huxley neuron under a constant external current.
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
        trace.time_ms.into_pyarray(py).unbind(),
        trace.voltage_mv.into_pyarray(py).unbind(),
        trace.spike_times_ms.into_pyarray(py).unbind(),
    ))
}

/// Simulate the NICE pathway with the grounded bilayer-sonophore capacitance
/// source (Plaksin et al. 2014, Eq. 8).
///
/// The acoustic carrier modulates the membrane capacitance through the leaflet
/// deflection; the displacement current drives a Hodgkin–Huxley membrane. The
/// trace exposes the membrane charge so the post-stimulus charge-accumulation
/// depolarisation can be inspected.
///
/// Args:
///     freq_mhz: Carrier frequency [MHz].
///     deflection_nm: Peak leaflet deflection Z_max [nm].
///     dt_ms: Integration step [ms] (must resolve the carrier; use ≤ period/50).
///     onset_ms: Sonication onset [ms].
///     offset_ms: Sonication offset [ms] (APs typically appear after this).
///     t_end_ms: Simulation duration [ms] (> offset_ms to see the response).
///     v_rest_mv: Resting potential [mV] (default −65).
///     i_bias_ua_cm2: Constant bias current density [µA/cm²] (default 0).
///     cm0_uf_cm2: Rest specific capacitance [µF/cm²] (default 1).
///
/// Returns:
///     (time_ms, voltage_mv, charge_nc_cm2, spike_times_ms) as numpy arrays.
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
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
)> {
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
        trace.time_ms.into_pyarray(py).unbind(),
        trace.voltage_mv.into_pyarray(py).unbind(),
        trace.charge_nc_cm2.into_pyarray(py).unbind(),
        trace.spike_times_ms.into_pyarray(py).unbind(),
    ))
}

/// Simulate the NICE pathway with the cycle-averaged SONIC reduction
/// (Lemaire et al. 2019).
///
/// Equivalent to `nice_bilayer_sonophore_response` but integrated in charge
/// density at the slow (millisecond) timescale, so it does not resolve the
/// acoustic carrier and scales to whole multi-second protocols. Reproduces the
/// carrier-resolved result for a single burst.
///
/// Args:
///     freq_mhz: Carrier frequency [MHz].
///     deflection_nm: Peak leaflet deflection Z_max [nm].
///     dt_ms: Slow integration step [ms] (e.g. 5e-3; need not resolve carrier).
///     onset_ms: Sonication onset [ms].
///     offset_ms: Sonication offset [ms].
///     t_end_ms: Simulation duration [ms].
///     v_rest_mv: Resting potential [mV] (default −65).
///     i_bias_ua_cm2: Constant bias current density [µA/cm²] (default 0).
///     cm0_uf_cm2: Rest specific capacitance [µF/cm²] (default 1).
///     cycle_samples: Samples per carrier cycle for averaging (default 64).
///
/// Returns:
///     (time_ms, voltage_mv, charge_nc_cm2, spike_times_ms) as numpy arrays.
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
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
)> {
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
        trace.time_ms.into_pyarray(py).unbind(),
        trace.voltage_mv.into_pyarray(py).unbind(),
        trace.charge_nc_cm2.into_pyarray(py).unbind(),
        trace.spike_times_ms.into_pyarray(py).unbind(),
    ))
}

/// NICE simulation with the **pressure-driven** quasi-static bilayer sonophore:
/// the leaflet deflection is solved from the acoustic pressure via the
/// intermolecular/elastic/electrical/gas force balance (not an assumed shape).
///
/// Args:
///     pressure_amp_pa: Acoustic peak pressure amplitude [Pa].
///     freq_mhz: Carrier frequency [MHz].
///     dt_ms: Integration step [ms] (resolve the carrier; ≤ period/50).
///     onset_ms, offset_ms, t_end_ms: Sonication window and duration [ms].
///     v_rest_mv: Resting potential [mV] (default −65; sets the resting charge).
///     i_bias_ua_cm2: Constant bias current density [µA/cm²] (default 0).
///     cm0_uf_cm2: Rest specific capacitance [µF/cm²] (default 1).
///
/// Returns:
///     (time_ms, voltage_mv, charge_nc_cm2, spike_times_ms) as numpy arrays.
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
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
)> {
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
        trace.time_ms.into_pyarray(py).unbind(),
        trace.voltage_mv.into_pyarray(py).unbind(),
        trace.charge_nc_cm2.into_pyarray(py).unbind(),
        trace.spike_times_ms.into_pyarray(py).unbind(),
    ))
}

/// NICE simulation with the **exact transient** bilayer sonophore: the leaflet
/// deflection is the steady-state solution of the full Rayleigh–Plesset ODE
/// (Plaksin Eq. 2, inertia + viscosity + gas), reproducing Plaksin Fig. 1
/// (≈ 12 nm peak deflection at 500 kPa / 0.5 MHz). The acoustic pressure is the
/// input; no quasi-static or kinematic approximation.
///
/// Args:
///     pressure_amp_pa: Acoustic peak pressure amplitude [Pa].
///     freq_mhz: Carrier frequency [MHz].
///     dt_ms: HH integration step [ms].
///     onset_ms, offset_ms, t_end_ms: Sonication window and duration [ms].
///     v_rest_mv: Resting potential [mV] (default −65).
///     i_bias_ua_cm2: Constant bias current density [µA/cm²] (default 0).
///     cm0_uf_cm2: Rest specific capacitance [µF/cm²] (default 1).
///
/// Returns:
///     (time_ms, voltage_mv, charge_nc_cm2, spike_times_ms) as numpy arrays.
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
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
)> {
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
        trace.time_ms.into_pyarray(py).unbind(),
        trace.voltage_mv.into_pyarray(py).unbind(),
        trace.charge_nc_cm2.into_pyarray(py).unbind(),
        trace.spike_times_ms.into_pyarray(py).unbind(),
    ))
}

/// Quasi-static leaflet deflection [m] versus acoustic pressure [Pa] (the
/// pressure→deflection relation from the bilayer-sonophore force balance).
///
/// Args:
///     pressure_pa: Acoustic pressure array [Pa] (sign: >0 compresses, <0 expands).
///     v_rest_mv: Resting potential [mV] (sets the resting charge).
///     cm0_uf_cm2: Rest specific capacitance [µF/cm²].
///
/// Returns:
///     Leaflet deflection array [m] (≥ 0; rectified at compression).
#[pyfunction]
#[pyo3(signature = (pressure_pa, v_rest_mv = -65.0, cm0_uf_cm2 = 1.0))]
pub fn bls_deflection_curve(
    py: Python<'_>,
    pressure_pa: PyReadonlyArray1<f64>,
    v_rest_mv: f64,
    cm0_uf_cm2: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let p = pressure_pa
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let qm0 = (cm0_uf_cm2 * 1.0e-2) * (v_rest_mv * 1.0e-3);
    let delta = rest_gap(qm0);
    let out: Vec<f64> = p
        .iter()
        .map(|&pac| quasistatic_deflection(pac, qm0, delta))
        .collect();
    Ok(out.into_pyarray(py).unbind())
}

/// SONIC simulation on a Pospischil cortical neuron (regular- or fast-spiking),
/// the membrane model the NICE framework actually uses — enables cell-type
/// selectivity.
///
/// Args:
///     neuron_class: "rs" (regular-spiking) or "fs" (fast-spiking).
///     freq_mhz: Carrier frequency [MHz].
///     deflection_nm: Peak leaflet deflection Z_max [nm].
///     dt_ms: Slow integration step [ms].
///     onset_ms, offset_ms, t_end_ms: Sonication window and duration [ms].
///     i_bias_ua_cm2: Constant bias current density [µA/cm²] (default 0).
///     cycle_samples: Samples per carrier cycle for averaging (default 64).
///
/// Returns:
///     (time_ms, voltage_mv, charge_nc_cm2, spike_times_ms) as numpy arrays.
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
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
)> {
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
        trace.time_ms.into_pyarray(py).unbind(),
        trace.voltage_mv.into_pyarray(py).unbind(),
        trace.charge_nc_cm2.into_pyarray(py).unbind(),
        trace.spike_times_ms.into_pyarray(py).unbind(),
    ))
}

/// Curved-dome bilayer membrane capacitance C_m(Z) (Plaksin Eq. 8) over an array
/// of leaflet deflections.
///
/// Args:
///     z_m: Leaflet deflection array [m].
///     cm0_uf_cm2: Rest specific capacitance [µF/cm²].
///     radius_a_m: Sonophore radius a [m].
///     gap_delta_m: Rest inter-leaflet gap Δ [m].
///
/// Returns:
///     Specific capacitance array [µF/cm²].
#[pyfunction]
#[pyo3(signature = (z_m, cm0_uf_cm2, radius_a_m, gap_delta_m))]
pub fn bilayer_capacitance_curve(
    py: Python<'_>,
    z_m: PyReadonlyArray1<f64>,
    cm0_uf_cm2: f64,
    radius_a_m: f64,
    gap_delta_m: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let z = z_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out: Vec<f64> = z
        .iter()
        .map(|&zi| bls_capacitance(zi, cm0_uf_cm2, radius_a_m, gap_delta_m))
        .collect();
    Ok(out.into_pyarray(py).unbind())
}

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

/// Minimum acoustic pressure amplitude [Pa] that evokes a post-stimulus action
/// potential, found by bisection with the quasi-static bilayer-sonophore source.
///
/// Returns `None` if no threshold lies within `[p_lo_pa, p_hi_pa]`.
///
/// Args:
///     neuron_class: "squid", "rs" (regular-spiking), or "fs" (fast-spiking).
///     freq_mhz: Carrier frequency [MHz].
///     p_lo_pa, p_hi_pa: Pressure search bracket [Pa].
///     n_iter: Bisection iterations (resolution ≈ (p_hi−p_lo)/2^n_iter).
///     dt_ms, onset_ms, offset_ms, t_end_ms: NICE integration window [ms].
///     cm0_uf_cm2: Rest specific capacitance [µF/cm²].
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

/// ITRUSST biophysical-safety assessment of a transcranial-US exposure
/// (Aubry et al. 2024 consensus reference levels).
///
/// Non-significant risk requires MI ≤ 1.9 and a thermal criterion (peak ΔT ≤ 2 °C
/// or brain CEM43 ≤ 2). These are informative consensus levels, not regulatory
/// limits.
///
/// Args:
///     mechanical_index: Peak MI [-].
///     peak_temp_rise_c: Peak focal temperature rise [°C].
///     cem43_brain_min: Cumulative brain thermal dose [CEM43 min].
///
/// Returns:
///     dict with mechanical_ok, thermal_ok, overall_ok.
#[pyfunction]
#[pyo3(signature = (mechanical_index, peak_temp_rise_c, cem43_brain_min))]
pub fn itrusst_safety(
    py: Python<'_>,
    mechanical_index: f64,
    peak_temp_rise_c: f64,
    cem43_brain_min: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let a = itrusst_assess(mechanical_index, peak_temp_rise_c, cem43_brain_min);
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("mechanical_ok", a.mechanical_ok)?;
    dict.set_item("thermal_ok", a.thermal_ok)?;
    dict.set_item("overall_ok", a.overall_ok)?;
    Ok(dict.unbind())
}

/// Pulse-train dosimetry for an ultrasonic-neuromodulation protocol
/// (Blackmore et al. 2019, Table 1).
///
/// Args:
///     carrier_freq_hz: Carrier frequency f [Hz].
///     pulse_length_s: Pulse length PL [s].
///     prf_hz: Pulse repetition frequency PRF [Hz].
///     burst_duration_s: Burst duration BD [s].
///     burst_interval_s: Burst interval BI [s].
///     num_bursts: Number of bursts N.
///     peak_pressure_pa: Carrier peak pressure [Pa].
///     density_kg_m3: Medium density [kg/m³].
///     sound_speed_m_s: Medium sound speed [m/s].
///
/// Returns:
///     dict with isppa_w_cm2, ispba_w_cm2, ispta_w_cm2, mechanical_index,
///     total_duty_cycle, total_time_s, within_fda_limits.
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
) -> PyResult<Py<pyo3::types::PyDict>> {
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
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("isppa_w_cm2", d.isppa_w_cm2)?;
    dict.set_item("ispba_w_cm2", d.ispba_w_cm2)?;
    dict.set_item("ispta_w_cm2", d.ispta_w_cm2)?;
    dict.set_item("mechanical_index", d.mechanical_index)?;
    dict.set_item("total_duty_cycle", d.total_duty_cycle)?;
    dict.set_item("total_time_s", d.total_time_s)?;
    dict.set_item("within_fda_limits", d.within_fda_limits())?;
    Ok(dict.unbind())
}
