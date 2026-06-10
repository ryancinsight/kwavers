//! Auto-organised from the original single-file `tests.rs` (split by concern).

use super::*;

#[test]
fn rate_singularities_use_lhopital_limits() {
    // α_m has a removable singularity at V = −40 (limit 1.0); α_n at V = −55
    // (limit 0.1). Verify the analytic limits and continuity across them.
    assert!(
        (alpha_m(-40.0) - 1.0).abs() < 1e-9,
        "α_m(-40) = {}",
        alpha_m(-40.0)
    );
    assert!(
        (alpha_n(-55.0) - 0.1).abs() < 1e-9,
        "α_n(-55) = {}",
        alpha_n(-55.0)
    );
    // Continuity: value at the singularity ≈ average of neighbours.
    let am_avg = 0.5 * (alpha_m(-40.0 + 1e-3) + alpha_m(-40.0 - 1e-3));
    assert!((alpha_m(-40.0) - am_avg).abs() < 1e-4);
    let an_avg = 0.5 * (alpha_n(-55.0 + 1e-3) + alpha_n(-55.0 - 1e-3));
    assert!((alpha_n(-55.0) - an_avg).abs() < 1e-4);
}

#[test]
fn resting_steady_state_matches_reference() {
    // Canonical HH steady-state gating at V = −65 mV (Dayan & Abbott 2001):
    // m∞ ≈ 0.053, h∞ ≈ 0.596, n∞ ≈ 0.318.
    let s = HhState::resting(V_REST);
    assert!((s.m - 0.0529).abs() < 2e-3, "m∞ = {}", s.m);
    assert!((s.h - 0.5961).abs() < 3e-3, "h∞ = {}", s.h);
    assert!((s.n - 0.3177).abs() < 3e-3, "n∞ = {}", s.n);
    // beta_h is a logistic bounded in (0,1).
    assert!(beta_h(0.0) > 0.0 && beta_h(0.0) < 1.0);
}

#[test]
fn resting_membrane_has_near_zero_net_current() {
    // E_L = −54.387 is chosen so the net ionic current vanishes at V ≈ −65 mV:
    // this is the defining condition of the resting potential.
    let p = HhParams::default();
    let i_ionic = HhState::resting(V_REST).ionic_current(&p);
    assert!(
        i_ionic.abs() < 0.05,
        "resting net current = {i_ionic} µA/cm²"
    );
}

#[test]
fn unforced_membrane_stays_at_rest() {
    let p = HhParams::default();
    let trace = simulate_hh(&p, V_REST, |_| 0.0, 0.01, 50.0);
    assert_eq!(trace.spike_count(), 0, "no spikes without drive");
    let max_dev = trace
        .voltage_mv
        .iter()
        .map(|&v| (v - V_REST).abs())
        .fold(0.0, f64::max);
    assert!(max_dev < 1.0, "rest drift {max_dev} mV");
}

#[test]
fn suprathreshold_step_fires_action_potentials() {
    // A sustained 15 µA/cm² step is well above rheobase (≈ 2.2 µA/cm²) and
    // elicits repetitive firing with the characteristic ≈ +40 mV overshoot.
    let p = HhParams::default();
    let trace = simulate_hh(&p, V_REST, |_| 15.0, 0.01, 50.0);
    assert!(trace.spike_count() >= 2, "spikes = {}", trace.spike_count());
    let v_max = trace.voltage_mv.iter().cloned().fold(f64::MIN, f64::max);
    assert!(v_max > 20.0, "AP overshoot only reached {v_max} mV");
    assert!(v_max < 60.0, "overshoot {v_max} mV exceeds E_Na bound");
    assert!(trace.mean_firing_rate_hz() > 0.0);
}

#[test]
fn subthreshold_step_does_not_fire() {
    let p = HhParams::default();
    let trace = simulate_hh(&p, V_REST, |_| 1.0, 0.01, 50.0);
    assert_eq!(trace.spike_count(), 0, "1 µA/cm² is below rheobase");
}

#[test]
fn firing_rate_increases_with_current() {
    // Above rheobase the HH f–I curve is monotone increasing.
    let p = HhParams::default();
    let n_low = simulate_hh(&p, V_REST, |_| 7.0, 0.01, 80.0).spike_count();
    let n_high = simulate_hh(&p, V_REST, |_| 20.0, 0.01, 80.0).spike_count();
    assert!(n_high > n_low, "f–I not monotone: {n_low} → {n_high}");
}

#[test]
fn cortical_rs_rests_near_canonical_potential() {
    // The RS neuron's net ionic current is small at its canonical resting
    // potential (−71.9 mV), and the M-current gate p∞ is low at rest.
    use super::super::cortical::CorticalNeuron;
    use super::super::membrane::Membrane;
    let rs = CorticalNeuron::regular_spiking();
    let g = rs.resting_gates(CorticalNeuron::V_REST_RS_MV);
    let i = rs.ionic_current(&g, CorticalNeuron::V_REST_RS_MV);
    assert!(i.abs() < 2.0, "RS resting net current {i} µA/cm²");
    assert!(g[3] > 0.0 && g[3] < 0.2, "M-gate p∞ at rest = {}", g[3]);
    assert!(g[0] < 0.1, "m∞ should be small at rest: {}", g[0]);
    assert!(g[1] > 0.5, "h∞ should be high at rest: {}", g[1]);
}

#[test]
fn cortical_fs_rests_near_canonical_potential() {
    // The fast-spiking interneuron preset is also near equilibrium at its
    // canonical rest (−71.4 mV) with the same quiescent gating signature, and it
    // differs from the RS preset (distinct conductances ⇒ distinct kinetics).
    use super::super::cortical::CorticalNeuron;
    use super::super::membrane::Membrane;
    let fs = CorticalNeuron::fast_spiking();
    let g = fs.resting_gates(CorticalNeuron::V_REST_FS_MV);
    let i = fs.ionic_current(&g, CorticalNeuron::V_REST_FS_MV);
    assert!(i.abs() < 2.0, "FS resting net current {i} µA/cm²");
    assert!(g[0] < 0.1 && g[1] > 0.5, "FS gating not quiescent: m={} h={}", g[0], g[1]);
    // RS and FS are genuinely different cells (FS has higher Na, lower Kd).
    let rs = CorticalNeuron::regular_spiking();
    assert!(fs.g_na_ms_cm2 != rs.g_na_ms_cm2 && fs.g_kd_ms_cm2 != rs.g_kd_ms_cm2);
}
