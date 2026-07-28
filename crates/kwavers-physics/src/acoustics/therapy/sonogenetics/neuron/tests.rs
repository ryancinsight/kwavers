use super::*;
use aequitas::systems::si::quantities::{Capacitance, ElectricCurrent, ElectricPotential, Time};
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
    let i_ion = ElectricCurrent::from_base(50.0e-12_f64);
    let dt = Time::from_base(0.1e-3_f64);
    let n_steps = 500;
    let mut t = Time::from_base(0.0);
    for _ in 0..n_steps {
        let spiked = neuron.step(i_ion, dt, t).unwrap();
        assert!(
            !spiked,
            "no spike expected for sub-threshold current at t={:.3e}",
            t.into_base()
        );
        t += dt;
    }
    let v_ss: ElectricPotential<f64> = params.leak_reversal + i_ion / params.leak_conductance;
    assert!(
        v_ss.into_base() < params.threshold.into_base(),
        "steady-state voltage {:.4e} must be below threshold {:.4e}",
        v_ss.into_base(),
        params.threshold.into_base()
    );
    assert_relative_eq!(
        neuron.membrane_voltage().into_base(),
        v_ss.into_base(),
        max_relative = 1e-3
    );
    assert_eq!(neuron.spike_count(), 0);
}

/// Supra-threshold constant current must produce spikes.
///
/// I = 200 pA > 100 pA threshold → repetitive firing.
#[test]
fn test_suprathreshold_produces_spikes() {
    let params = LifParams::default();
    let mut neuron = LifNeuron::new(params);
    let i_ion = ElectricCurrent::from_base(200.0e-12_f64);
    let dt = Time::from_base(0.05e-3_f64);
    let duration = Time::from_base(100.0e-3_f64);
    let n_steps = (duration.into_base() / dt.into_base()) as usize;
    let mut t = Time::from_base(0.0);
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
    let i_large = ElectricCurrent::from_base(1.0e-9_f64);
    let dt = Time::from_base(0.1e-3_f64);
    let mut t = Time::from_base(0.0);
    let mut spiked_once = false;
    for _ in 0..200 {
        let spiked = neuron.step(i_large, dt, t).unwrap();
        t += dt;
        if spiked {
            spiked_once = true;
            assert_relative_eq!(
                neuron.membrane_voltage().into_base(),
                params.reset.into_base(),
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
    assert!(neuron
        .step(
            ElectricCurrent::from_base(0.0),
            Time::from_base(0.0),
            Time::from_base(0.0),
        )
        .is_err());
    assert!(neuron
        .step(
            ElectricCurrent::from_base(0.0),
            Time::from_base(-1e-6),
            Time::from_base(0.0),
        )
        .is_err());
}

/// Mean firing rate is spike_count / duration.
#[test]
fn test_mean_firing_rate() {
    let params = LifParams::default();
    let mut neuron = LifNeuron::new(params);
    let i_ion = ElectricCurrent::from_base(200.0e-12_f64);
    let dt = Time::from_base(0.05e-3_f64);
    let duration = Time::from_base(100.0e-3_f64);
    let n_steps = (duration.into_base() / dt.into_base()) as usize;
    let mut t = Time::from_base(0.0);
    for _ in 0..n_steps {
        let _ = neuron.step(i_ion, dt, t).unwrap();
        t += dt;
    }
    let rate = neuron.mean_firing_rate(duration);
    let expected = neuron.spike_count() as f64 / duration.into_base();
    assert_relative_eq!(rate.into_base(), expected, max_relative = 1e-12);
    assert_eq!(
        neuron.mean_firing_rate(Time::from_base(0.0)).into_base(),
        0.0
    );
    assert_eq!(
        neuron.mean_firing_rate(Time::from_base(-1.0)).into_base(),
        0.0
    );
}

/// Membrane time constant equals C_m / G_leak.
#[test]
fn test_time_constant() {
    let params = LifParams::default();
    let tau = params.time_constant();
    assert_relative_eq!(tau.into_base(), 10.0e-3, max_relative = 1e-12);
}

/// LifParams validity check.
#[test]
fn test_params_validity() {
    let valid = LifParams::default();
    assert!(valid.is_valid());
    let bad = LifParams {
        capacitance: Capacitance::from_base(0.0),
        ..Default::default()
    };
    assert!(!bad.is_valid());
    let mut bad2 = LifParams::default();
    bad2.threshold = bad2.reset - ElectricPotential::from_base(1e-3);
    assert!(!bad2.is_valid());
}
