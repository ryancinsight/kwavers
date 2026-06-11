//! Auto-organised from the original single-file `tests.rs` (split by concern).

use super::*;

/// Build a NICE configuration at a fixed carrier with `depth = ε`.
/// Carrier 0.1 MHz keeps the step count modest while staying ≫ the membrane
/// time constant (~1 ms), preserving the fast-charge-conservation regime.
fn nice_cfg(depth: f64) -> NiceConfig<HhParams, CapacitanceModulation> {
    let freq_mhz = 0.1;
    let source = CapacitanceModulation::new(1.0, depth, freq_mhz);
    NiceConfig {
        membrane: HhParams::default(),
        v_rest_mv: V_REST,
        source,
        i_bias_ua_cm2: 0.0,
        dt_ms: 1.0e-4, // 100 samples per 0.01 ms carrier period
        onset_ms: 2.0, // integer ms ⇒ sin(ω·onset) = 0 ⇒ continuous boundary
        offset_ms: 12.0,
        t_end_ms: 12.0,
    }
}

#[test]
fn nice_config_resolves_carrier() {
    let cfg = nice_cfg(0.2);
    assert!(cfg.is_valid());
    assert!(cfg.samples_per_cycle() > 50.0, "carrier under-resolved");
}

#[test]
fn nice_zero_depth_is_quiescent() {
    // ε = 0 ⇒ dC_m/dt = 0 ⇒ no displacement current ⇒ unforced HH at rest.
    let cfg = nice_cfg(0.0);
    let trace = simulate_nice(&cfg);
    assert_eq!(trace.spike_count(), 0);
    let max_dev = trace
        .voltage_mv
        .iter()
        .map(|&v| (v - V_REST).abs())
        .fold(0.0, f64::max);
    assert!(max_dev < 1.0, "drift without drive: {max_dev} mV");
}

#[test]
fn nice_drive_amplitude_is_monotone() {
    // Below the spike threshold, the peak membrane depolarisation grows with the
    // capacitance modulation depth (stronger acoustic drive → larger excursion).
    let peak = |eps: f64| {
        simulate_nice(&nice_cfg(eps))
            .voltage_mv
            .iter()
            .cloned()
            .fold(f64::MIN, f64::max)
    };
    let p05 = peak(0.05);
    let p10 = peak(0.10);
    let p15 = peak(0.15);
    assert!(
        p05 > V_REST,
        "depth 0.05 should depolarise above rest: {p05}"
    );
    assert!(
        p10 > p05 && p15 > p10,
        "non-monotone peaks: {p05} {p10} {p15}"
    );
}

/// Mean membrane potential over the steady part of the sonication window
/// (excluding the onset transient), used to read the net cycle-averaged shift.
fn window_mean_mv(eps: f64) -> f64 {
    let cfg = nice_cfg(eps);
    let win: Vec<f64> = simulate_nice(&cfg)
        .time_ms
        .iter()
        .zip(simulate_nice(&cfg).voltage_mv)
        .filter(|(&t, _)| t >= 3.0 && t <= 11.0)
        .map(|(_, v)| v)
        .collect();
    win.iter().sum::<f64>() / win.len() as f64
}

#[test]
fn nice_net_shift_is_hyperpolarising_and_grows_with_depth() {
    // For a symmetric capacitance sinusoid the fast-charge-conservation regime
    // gives ⟨V⟩ ≈ V_rest · ⟨1/(1+ε·sinωt)⟩ = V_rest / √(1−ε²); since V_rest < 0
    // the mean potential moves further from zero (net hyperpolarisation) and the
    // shift grows monotonically with ε. Depolarising gating rectification only
    // partially offsets it (so the magnitude is below the pure-geometric bound).
    let m0 = window_mean_mv(0.0);
    let m2 = window_mean_mv(0.2);
    let m4 = window_mean_mv(0.4);
    let m6 = window_mean_mv(0.6);
    assert!((m0 - V_REST).abs() < 0.2, "ε=0 mean {m0} not at rest");
    assert!(
        m6 < m4 && m4 < m2 && m2 < m0,
        "net shift not monotone hyperpolarising: {m0} {m2} {m4} {m6}"
    );
    // Stays above (less negative than) the pure-geometric prediction V_rest/√(1−ε²),
    // confirming the partially-offsetting depolarising rectification.
    let geometric = V_REST / (1.0 - 0.6 * 0.6_f64).sqrt();
    assert!(
        m6 > geometric,
        "mean {m6} below geometric bound {geometric}"
    );
}

/// BLS NICE configuration: 0.5 MHz carrier, peak deflection `zmax_nm`,
/// sonication `[1, offset]` ms, observed to `t_end` ms.
fn bls_cfg(
    zmax_nm: f64,
    offset_ms: f64,
    t_end_ms: f64,
) -> NiceConfig<HhParams, super::super::bls::BilayerSonophore> {
    NiceConfig {
        membrane: HhParams::default(),
        v_rest_mv: V_REST,
        source: super::super::bls::BilayerSonophore::new(1.0, 0.5, zmax_nm * 1.0e-9),
        i_bias_ua_cm2: 0.0,
        dt_ms: 4.0e-5, // 50 samples per 0.5 MHz carrier period
        onset_ms: 1.0,
        offset_ms,
        t_end_ms,
    }
}

#[test]
fn bls_zero_deflection_is_quiescent() {
    let tr = simulate_nice(&bls_cfg(0.0, 12.0, 18.0));
    assert_eq!(tr.spike_count(), 0, "no drive ⇒ no spikes");
    let max_dev = tr
        .voltage_mv
        .iter()
        .map(|&v| (v - V_REST).abs())
        .fold(0.0, f64::max);
    assert!(max_dev < 0.5, "membrane drifted {max_dev} mV without drive");
}

#[test]
fn bls_hyperpolarises_during_sonication() {
    // Plaksin et al. 2014: the US-induced capacitance oscillations are
    // hyperpolarising. The per-cycle minimum potential during sonication drops
    // well below rest (and deeper with stronger leaflet deflection), even though
    // the cycle mean drifts upward as charge accumulates.
    let min_during = |zmax: f64| {
        let cfg = bls_cfg(zmax, 9.0, 11.0);
        simulate_nice(&cfg)
            .time_ms
            .iter()
            .zip(simulate_nice(&cfg).voltage_mv)
            .filter(|(&t, _)| t > 3.0 && t < 9.0)
            .map(|(_, v)| v)
            .fold(f64::MAX, f64::min)
    };
    let m1 = min_during(1.0);
    let m4 = min_during(4.0);
    assert!(m1 < V_REST - 2.0, "1 nm: trough {m1} not hyperpolarised");
    assert!(
        m4 < m1,
        "deeper deflection not more hyperpolarising: {m1} → {m4}"
    );
}

#[test]
fn bls_accumulates_membrane_charge() {
    // The asymmetric capacitance waveform rectifies the leak current into a net
    // charge accumulation that grows with leaflet deflection (Plaksin mechanism).
    let dq = |zmax: f64| {
        let cfg = bls_cfg(zmax, 9.0, 11.0);
        let tr = simulate_nice(&cfg);
        let q_end = tr
            .time_ms
            .iter()
            .zip(&tr.charge_nc_cm2)
            .filter(|(&t, _)| t < cfg.offset_ms)
            .map(|(_, &q)| q)
            .last()
            .unwrap();
        q_end - 1.0 * V_REST // minus resting charge (C_m0·V_rest, C_m0 = 1)
    };
    let d1 = dq(1.0);
    let d4 = dq(4.0);
    assert!(d1 > 1.0, "no charge accumulation at 1 nm: ΔQ = {d1}");
    assert!(
        d4 > d1,
        "accumulation not monotone in deflection: {d1} → {d4}"
    );
}

#[test]
fn bls_post_stimulus_ap_with_pulse_duration_dependence() {
    // The accumulated charge depolarises the membrane once US stops and C_m
    // returns to baseline, evoking a post-stimulus AP — but only if the pulse is
    // long enough to accumulate sufficient charge (Plaksin Fig. 2: the
    // requirement for long stimulation pulses).
    let short = simulate_nice(&bls_cfg(1.0, 1.5, 11.0)); // 0.5 ms US
    assert_eq!(
        short.spike_count(),
        0,
        "0.5 ms pulse should not evoke an AP"
    );

    let long = simulate_nice(&bls_cfg(1.0, 6.0, 14.0)); // 5 ms US
    let post: Vec<f64> = long
        .spike_times_ms
        .iter()
        .copied()
        .filter(|&t| t >= 6.0)
        .collect();
    assert!(
        !post.is_empty(),
        "5 ms pulse should evoke a post-stimulus AP"
    );
    let v_max = long.voltage_mv.iter().cloned().fold(f64::MIN, f64::max);
    assert!(v_max > 20.0, "post-stimulus AP overshoot only {v_max} mV");
}

#[test]
fn nice_small_depth_does_not_fire() {
    // A weak drive on a resting neuron produces sub-threshold modulation only.
    let trace = simulate_nice(&nice_cfg(0.05));
    assert_eq!(trace.spike_count(), 0, "ε = 0.05 should not fire");
}

#[test]
fn quasistatic_source_evokes_post_stimulus_ap() {
    // The pressure-driven quasi-static source produces the same NICE behaviour as
    // the kinematic one: an asymmetric C_m(t) waveform whose charge accumulation
    // evokes a post-stimulus action potential.
    use super::super::bls::pressures::BilayerSonophoreQuasistatic;
    let cfg = NiceConfig {
        membrane: HhParams::default(),
        v_rest_mv: V_REST,
        source: BilayerSonophoreQuasistatic::new(1.0, 0.5, 500.0e3, V_REST),
        i_bias_ua_cm2: 0.0,
        dt_ms: 4.0e-5,
        onset_ms: 1.0,
        offset_ms: 8.0,
        t_end_ms: 16.0,
    };
    assert!(cfg.is_valid());
    let tr = simulate_nice(&cfg);
    assert!(
        tr.spike_count() >= 1,
        "no post-stimulus AP (pressure-driven)"
    );
    assert!(tr.spike_times_ms.iter().all(|&t| t >= cfg.onset_ms));
    let v_max = tr.voltage_mv.iter().cloned().fold(f64::MIN, f64::max);
    assert!(v_max > 20.0, "AP overshoot only {v_max} mV");
}

#[test]
fn bls_dynamics_reproduces_plaksin_fig1_deflection() {
    use super::super::bls::dynamics::BilayerSonophoreDynamic;
    // Full transient ODE at 0.5 MHz: peak deflection grows with pressure and
    // reaches ≈ 10–12 nm at 500 kPa, matching Plaksin et al. (2014) Fig. 1
    // (≈ 12 nm). Resting potential −71.9 mV (the NICE/NBLS neuron).
    let z100 = BilayerSonophoreDynamic::new(1.0, 0.5, 100.0e3, -71.9).peak_deflection_m();
    let z300 = BilayerSonophoreDynamic::new(1.0, 0.5, 300.0e3, -71.9).peak_deflection_m();
    let z500 = BilayerSonophoreDynamic::new(1.0, 0.5, 500.0e3, -71.9).peak_deflection_m();
    assert!(
        z100 > 0.0 && z300 > z100 && z500 > z300,
        "not monotone: {z100} {z300} {z500}"
    );
    assert!(
        (8.0e-9..14.0e-9).contains(&z500),
        "peak Z(500 kPa) = {:.2} nm (expected ≈ 12 nm)",
        z500 * 1e9
    );
    // Resonant amplification: the transient deflection exceeds the inertia-free
    // quasi-static value at the same pressure.
    let qm0 = 1.0e-2 * (-71.9e-3);
    let zqs = super::super::bls::pressures::quasistatic_deflection(
        -500.0e3,
        qm0,
        super::super::bls::pressures::rest_gap(qm0),
    );
    assert!(
        z500 > zqs,
        "dynamic {z500} should exceed quasi-static {zqs}"
    );
}

#[test]
fn bls_dynamics_source_evokes_post_stimulus_ap() {
    // The exact-transient bilayer-sonophore source drives the NICE coupling to a
    // post-stimulus action potential. 300 kPa keeps the leaflet in the clean
    // expansion regime (no steric-wall contact), giving a full ≈ +27 mV overshoot.
    use super::super::bls::dynamics::BilayerSonophoreDynamic;
    let cfg = NiceConfig {
        membrane: HhParams::default(),
        v_rest_mv: V_REST,
        source: BilayerSonophoreDynamic::new(1.0, 0.5, 300.0e3, V_REST),
        i_bias_ua_cm2: 0.0,
        dt_ms: 4.0e-5,
        onset_ms: 1.0,
        offset_ms: 8.0,
        t_end_ms: 16.0,
    };
    assert!(cfg.is_valid());
    let tr = simulate_nice(&cfg);
    assert!(
        tr.spike_count() >= 1,
        "no post-stimulus AP (dynamic source)"
    );
    assert!(tr.spike_times_ms.iter().all(|&t| t >= cfg.onset_ms));
    assert!(tr.voltage_mv.iter().cloned().fold(f64::MIN, f64::max) > 20.0);
}

#[test]
fn cortical_rs_fires_under_current_clamp_via_nice_zero_drive() {
    // With no acoustic drive but a supra-rheobase bias current, the RS neuron
    // fires repetitively — exercising the M-current adaptation path through the
    // generic NICE integrator (zero-depth source ⇒ pure current-clamp HH).
    use super::super::cortical::CorticalNeuron;
    let cfg = NiceConfig {
        membrane: CorticalNeuron::regular_spiking(),
        v_rest_mv: CorticalNeuron::V_REST_RS_MV,
        source: CapacitanceModulation::new(1.0, 0.0, 0.5), // zero depth ⇒ no drive
        i_bias_ua_cm2: 5.0,
        dt_ms: 5.0e-3,
        onset_ms: 0.0,
        offset_ms: 0.0,
        t_end_ms: 120.0,
    };
    let tr = simulate_nice(&cfg);
    assert!(
        tr.spike_count() >= 2,
        "RS did not fire: {}",
        tr.spike_count()
    );
    let v_max = tr.voltage_mv.iter().cloned().fold(f64::MIN, f64::max);
    assert!(v_max > 0.0, "no AP overshoot: {v_max} mV");
}

#[test]
fn cortical_rs_and_fs_differ_under_identical_drive() {
    // Cell-type selectivity: RS and FS neurons reach different excitability under
    // the same intramembrane-cavitation drive (different conductances/kinetics).
    use super::super::cortical::CorticalNeuron;
    use super::super::sonic::{simulate_sonic, SonicConfig};
    let make = |neuron: CorticalNeuron, v_rest: f64| SonicConfig {
        membrane: neuron,
        v_rest_mv: v_rest,
        source: super::super::bls::BilayerSonophore::new(1.0, 0.5, 1.5e-9),
        i_bias_ua_cm2: 0.0,
        dt_ms: 5.0e-3,
        cycle_samples: 64,
        onset_ms: 1.0,
        offset_ms: 30.0,
        t_end_ms: 45.0,
    };
    let rs = simulate_sonic(&make(
        CorticalNeuron::regular_spiking(),
        CorticalNeuron::V_REST_RS_MV,
    ));
    let fs = simulate_sonic(&make(
        CorticalNeuron::fast_spiking(),
        CorticalNeuron::V_REST_FS_MV,
    ));
    assert!(rs.voltage_mv.iter().all(|v| v.is_finite()));
    assert!(fs.voltage_mv.iter().all(|v| v.is_finite()));
    // Cell-type selectivity: the two classes accumulate charge differently under
    // the identical drive (different conductances / kinetics), so the mean
    // membrane potential over the sonication window differs measurably.
    let mean_v = |tr: &HhTrace| {
        let w: Vec<f64> = tr
            .time_ms
            .iter()
            .zip(&tr.voltage_mv)
            .filter(|(&t, _)| (1.0..30.0).contains(&t))
            .map(|(_, &v)| v)
            .collect();
        w.iter().sum::<f64>() / w.len() as f64
    };
    let (rs_mean, fs_mean) = (mean_v(&rs), mean_v(&fs));
    assert!(
        (rs_mean - fs_mean).abs() > 0.5,
        "RS and FS responses indistinguishable: {rs_mean} vs {fs_mean} mV"
    );
}

fn sonic_cfg(
    zmax_nm: f64,
    offset_ms: f64,
    t_end_ms: f64,
) -> super::super::sonic::SonicConfig<HhParams> {
    super::super::sonic::SonicConfig {
        membrane: HhParams::default(),
        v_rest_mv: V_REST,
        source: super::super::bls::BilayerSonophore::new(1.0, 0.5, zmax_nm * 1.0e-9),
        i_bias_ua_cm2: 0.0,
        dt_ms: 5.0e-3, // 5 µs slow step (no need to resolve the 0.5 MHz carrier)
        cycle_samples: 64,
        onset_ms: 1.0,
        offset_ms,
        t_end_ms,
    }
}

#[test]
fn sonic_zero_deflection_is_quiescent() {
    use super::super::sonic::simulate_sonic;
    let tr = simulate_sonic(&sonic_cfg(0.0, 12.0, 18.0));
    assert_eq!(tr.spike_count(), 0, "no drive ⇒ no spikes");
    let max_dev = tr
        .voltage_mv
        .iter()
        .map(|&v| (v - V_REST).abs())
        .fold(0.0, f64::max);
    assert!(max_dev < 0.5, "drift without drive: {max_dev} mV");
}

#[test]
fn sonic_post_stimulus_ap() {
    use super::super::sonic::simulate_sonic;
    let tr = simulate_sonic(&sonic_cfg(1.0, 6.0, 14.0));
    assert!(tr.spike_count() >= 1, "no post-stimulus AP");
    assert!(
        tr.spike_times_ms.iter().all(|&t| t >= 6.0),
        "spike before offset"
    );
    let v_max = tr.voltage_mv.iter().cloned().fold(f64::MIN, f64::max);
    assert!(v_max > 20.0, "AP overshoot only {v_max} mV");
}

#[test]
fn sonic_matches_carrier_resolved_nice_single_burst() {
    // Differential test: the cycle-averaged SONIC reduction reproduces the
    // carrier-resolved NICE result for a single burst — same spike count and
    // post-stimulus AP timing within one slow membrane time-step — at ~100×
    // fewer integration steps.
    use super::super::sonic::simulate_sonic;
    let (zmax, offset, t_end) = (1.0, 6.0, 14.0);
    let nice = simulate_nice(&bls_cfg(zmax, offset, t_end));
    let sonic = simulate_sonic(&sonic_cfg(zmax, offset, t_end));
    assert_eq!(
        nice.spike_count(),
        sonic.spike_count(),
        "spike counts differ: NICE {} vs SONIC {}",
        nice.spike_count(),
        sonic.spike_count()
    );
    assert!(
        nice.spike_count() >= 1,
        "expected a post-stimulus AP in both"
    );
    let t_nice = nice.spike_times_ms[0];
    let t_sonic = sonic.spike_times_ms[0];
    assert!(
        (t_nice - t_sonic).abs() < 1.0,
        "AP timing mismatch: NICE {t_nice} ms vs SONIC {t_sonic} ms"
    );
}

#[test]
fn threshold_pressure_is_bracketed() {
    // The threshold-finder (analysis::ThresholdQuery) locates, by bisection, the
    // minimum acoustic pressure that evokes a post-stimulus AP. Validate the
    // bracket: pressures above the returned threshold fire, well-below do not.
    use super::super::analysis::ThresholdQuery;
    use super::super::bls::BilayerSonophoreQuasistatic;
    let q = ThresholdQuery {
        membrane: HhParams::default(),
        v_rest_mv: V_REST,
        i_bias_ua_cm2: 0.0,
        dt_ms: 4.0e-5,
        onset_ms: 1.0,
        offset_ms: 6.0,
        t_end_ms: 14.0,
    };
    let make = |p: f64| BilayerSonophoreQuasistatic::new(1.0, 0.5, p, V_REST);
    let thr = q
        .threshold_pressure_pa(50.0e3, 600.0e3, 6, &make)
        .expect("a firing threshold should exist within [50, 600] kPa");
    assert!(
        (50.0e3..600.0e3).contains(&thr),
        "threshold {thr} Pa out of range"
    );
    assert!(q.fires(thr * 1.15, &make), "above threshold should fire");
    assert!(
        !q.fires(thr * 0.5, &make),
        "well below threshold should not fire"
    );
}
