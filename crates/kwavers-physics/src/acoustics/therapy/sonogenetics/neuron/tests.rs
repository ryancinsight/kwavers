use super::*;
use eunomia::assert_relative_eq;

/// Sub-threshold constant current must not produce spikes.
///
/// Steady-state voltage for constant current I:
///   V_ss = E_leak + I / G_leak
/// Require: V_ss < V_thresh
///   I < G_leak · (V_thresh − E_leak) = 10e-9 × 10e-3 = 100 pA
#[test]
fn test_subthreshold_no_spike() {
    let params = LifParams::default();
    let mut neuron = LifNeuron::new(params.clone());
    let i_ion = 50.0e-12_f64;
    let dt = 0.1e-3_f64;
    let n_steps = 500;
    let mut t = 0.0;
    for _ in 0..n_steps {
        let spiked = neuron.step(i_ion, dt, t).unwrap();
        assert!(
            !spiked,
            "no spike expected for sub-threshold current at t={t:.3e}"
        );
        t += dt;
    }
    let v_ss = params.leak_reversal_v + i_ion / params.leak_conductance_s;
    assert!(
        v_ss < params.threshold_v,
        "steady-state voltage {v_ss:.4e} must be below threshold {:.4e}",
        params.threshold_v
    );
    assert_relative_eq!(neuron.membrane_voltage(), v_ss, max_relative = 1e-3);
    assert_eq!(neuron.spike_count(), 0);
}

/// Supra-threshold constant current must produce spikes.
///
/// I = 200 pA > 100 pA threshold → repetitive firing.
#[test]
fn test_suprathreshold_produces_spikes() {
    let params = LifParams::default();
    let mut neuron = LifNeuron::new(params);
    let i_ion = 200.0e-12_f64;
    let dt = 0.05e-3_f64;
    let duration = 100.0e-3_f64;
    let n_steps = (duration / dt) as usize;
    let mut t = 0.0;
    for _ in 0..n_steps {
        let _ = neuron.step(i_ion, dt, t).unwrap();
        t += dt;
    }
    assert!(
        neuron.spike_count() >= 3,
        "expected ≥3 spikes for I=200 pA over 100 ms, got {}",
        neuron.spike_count()
    );
}

/// After a spike, voltage must be at V_reset.
#[test]
fn test_refractory_clamp() {
    let params = LifParams::default();
    let mut neuron = LifNeuron::new(params.clone());
    let i_large = 1.0e-9_f64;
    let dt = 0.1e-3_f64;
    let mut t = 0.0;
    let mut spiked_once = false;
    for _ in 0..200 {
        let spiked = neuron.step(i_large, dt, t).unwrap();
        t += dt;
        if spiked {
            spiked_once = true;
            assert_relative_eq!(
                neuron.membrane_voltage(),
                params.reset_v,
                max_relative = 1e-9
            );
            break;
        }
    }
    assert!(
        spiked_once,
        "should have spiked with I = 1 nA over 200 steps"
    );
}

/// Zero time step returns an error.
#[test]
fn test_zero_dt_is_error() {
    let mut neuron = LifNeuron::new(LifParams::default());
    assert!(neuron.step(0.0, 0.0, 0.0).is_err());
    assert!(neuron.step(0.0, -1e-6, 0.0).is_err());
}

/// Mean firing rate is spike_count / duration.
#[test]
fn test_mean_firing_rate() {
    let params = LifParams::default();
    let mut neuron = LifNeuron::new(params);
    let i_ion = 200.0e-12_f64;
    let dt = 0.05e-3_f64;
    let duration = 100.0e-3_f64;
    let n_steps = (duration / dt) as usize;
    let mut t = 0.0;
    for _ in 0..n_steps {
        let _ = neuron.step(i_ion, dt, t).unwrap();
        t += dt;
    }
    let rate = neuron.mean_firing_rate(duration);
    let expected = neuron.spike_count() as f64 / duration;
    assert_relative_eq!(rate, expected, max_relative = 1e-12);
    assert_eq!(neuron.mean_firing_rate(0.0), 0.0);
    assert_eq!(neuron.mean_firing_rate(-1.0), 0.0);
}

/// Membrane time constant equals C_m / G_leak.
#[test]
fn test_time_constant() {
    let params = LifParams::default();
    let tau = params.time_constant_s();
    assert_relative_eq!(tau, 10.0e-3, max_relative = 1e-12);
}

/// LifParams validity check.
#[test]
fn test_params_validity() {
    let valid = LifParams::default();
    assert!(valid.is_valid());
    let bad = LifParams {
        capacitance_f: 0.0,
        ..Default::default()
    };
    assert!(!bad.is_valid());
    let mut bad2 = LifParams::default();
    bad2.threshold_v = bad2.reset_v - 1e-3;
    assert!(!bad2.is_valid());
}
