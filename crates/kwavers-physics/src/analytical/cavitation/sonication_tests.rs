use super::*;

#[test]
fn lacuna_void_growth_saturates() {
    let bmax = 5e-3;
    assert_eq!(lacuna_void_fraction(0.0, 1e9, 120.0, bmax), 0.0); // unlesioned
    assert!(lacuna_void_fraction(1.0, 0.0, 120.0, bmax).abs() < 1e-12); // t=0 no void
    let mid = lacuna_void_fraction(1.0, 120.0, 120.0, bmax);
    let inf = lacuna_void_fraction(1.0, 1e6, 120.0, bmax);
    assert!((inf - bmax).abs() < 1e-9, "saturates to beta_max: {inf}");
    assert!(mid > 0.0 && mid < inf); // monotone growth
                                     // scales with fractionation
    assert!((lacuna_void_fraction(0.5, 1e6, 120.0, bmax) - 0.5 * bmax).abs() < 1e-9);
}

#[test]
fn kill_fraction_dose_response_and_inverse() {
    let (d0, k) = (1.0, 2.5);
    // No dose → no kill; characteristic dose → 1 − 1/e.
    assert_eq!(histotripsy_kill_fraction(0.0, d0, k), 0.0);
    assert!((histotripsy_kill_fraction(d0, d0, k) - (1.0 - (-1.0f64).exp())).abs() < 1e-9);
    // Monotone increasing and bounded in [0,1].
    assert!(histotripsy_kill_fraction(0.5, d0, k) < histotripsy_kill_fraction(1.5, d0, k));
    assert!(histotripsy_kill_fraction(1e6, d0, k) <= 1.0);
    // LD_x inverts the dose-response: kill(LD_x) == x.
    for x in [0.25, 0.5, 0.75, 0.9] {
        let ld = histotripsy_lethal_dose(x, d0, k);
        assert!(
            (histotripsy_kill_fraction(ld, d0, k) - x).abs() < 1e-6,
            "LD{x}"
        );
    }
    // Iso-lethal levels are ordered LD25 < LD50 < LD75.
    let (a, b, c) = (
        histotripsy_lethal_dose(0.25, d0, k),
        histotripsy_lethal_dose(0.50, d0, k),
        histotripsy_lethal_dose(0.75, d0, k),
    );
    assert!(a < b && b < c);
}

#[test]
fn interface_enhancement_bounds() {
    // Matched media → no enhancement.
    assert!((interface_pressure_enhancement(1.65e6, 1.65e6) - 1.0).abs() < 1e-12);
    // Soft-tissue (liver/fat) interface is mild.
    let e_soft = interface_pressure_enhancement(1.65e6, 1.38e6);
    assert!(e_soft > 1.0 && e_soft < 1.15, "liver/fat E={e_soft}");
    // Tissue/gas (lacuna) interface approaches the pressure-doubling limit.
    let e_gas = interface_pressure_enhancement(1.65e6, 0.0004e6);
    assert!(e_gas > 1.95 && e_gas <= 2.0, "tissue/gas E={e_gas}");
}

#[test]
fn lacuna_susceptibility_delay_and_limits() {
    // Unlesioned tissue is unaffected.
    assert!((lacuna_cavitation_susceptibility(0.0, 1e9, 120.0, 0.5, 4.0) - 1.0).abs() < 1e-12);
    // Immediately after lesioning: only the prompt term (no lacuna yet).
    let s0 = lacuna_cavitation_susceptibility(1.0, 0.0, 120.0, 0.5, 4.0);
    assert!((s0 - 1.5).abs() < 1e-9, "prompt S={s0}");
    // Long after lesioning: full lacuna enhancement.
    let s_inf = lacuna_cavitation_susceptibility(1.0, 1e6, 120.0, 0.5, 4.0);
    assert!((s_inf - 5.5).abs() < 1e-3, "lacuna S={s_inf}");
    // Monotonic growth with elapsed time.
    let s_mid = lacuna_cavitation_susceptibility(1.0, 120.0, 120.0, 0.5, 4.0);
    assert!(s0 < s_mid && s_mid < s_inf);
}

#[test]
fn interleaved_schedule_timing_matches_diagram() {
    // 3 sub-spots, 4 repetitions, 10 µs pulse, 5 Hz fired-pulse rate.
    let n = 3usize;
    let n_rep = 4usize;
    let pd = 10e-6;
    let prf = 5.0;
    let s = build_sonication_schedule(n, n_rep, pd, prf, SonicationOrder::Interleaved);
    assert_eq!(s.onset_s.len(), n * n_rep);
    // First repetition fires sub-spots 0,1,2 at 0, 0.2, 0.4 s.
    assert_eq!(&s.subspot[0..3], &[0, 1, 2]);
    assert!((s.onset_s[1] - 0.2).abs() < 1e-12);
    // Repetition time = one grid pass = n/PRF = 0.6 s.
    assert!((s.repetition_time_s - n as f64 / prf).abs() < 1e-12);
    // Sub-spot 0 fires once per repetition → effective per-spot interval = 0.6 s.
    let spot0: Vec<f64> = s
        .onset_s
        .iter()
        .zip(&s.subspot)
        .filter(|(_, &sp)| sp == 0)
        .map(|(&t, _)| t)
        .collect();
    assert_eq!(spot0.len(), n_rep);
    assert!((spot0[1] - spot0[0] - 0.6).abs() < 1e-12);
    // Sonication duration = last onset + pulse duration.
    let last = (n * n_rep - 1) as f64 / prf;
    assert!((s.sonication_duration_s - (last + pd)).abs() < 1e-12);
}

#[test]
fn sequential_schedule_fires_one_spot_at_a_time() {
    let n = 3usize;
    let n_rep = 4usize;
    let s = build_sonication_schedule(n, n_rep, 10e-6, 5.0, SonicationOrder::Sequential);
    // First n_rep pulses are all sub-spot 0.
    assert!(s.subspot[0..n_rep].iter().all(|&sp| sp == 0));
    assert_eq!(s.subspot[n_rep], 1);
    // Sequential per-spot dwell = n_rep/PRF.
    assert!((s.repetition_time_s - n_rep as f64 / 5.0).abs() < 1e-12);
}

#[test]
fn delivery_fraction_decreases_with_gas_and_attenuation() {
    // Air-in-water residual cloud at 0.5 MHz, 2 µm bubbles.
    let base = forward_delivery_fraction(
        1.0, 1.5e6, 1.5e6, 5.0, 0.05, 0.0, 0.5e6, 2e-6, 1481.0, 998.0, 1e-3, 101_325.0, 1.4,
    );
    let with_gas = forward_delivery_fraction(
        1.0, 1.5e6, 1.5e6, 5.0, 0.05, 1e-4, 0.5e6, 2e-6, 1481.0, 998.0, 1e-3, 101_325.0, 1.4,
    );
    assert!(base > 0.0 && base <= 1.0);
    assert!(
        with_gas < base,
        "residual gas must reduce delivered pressure: {with_gas} < {base}"
    );
}

#[test]
fn received_fraction_is_two_way_loss() {
    // With no gas and matched impedance, the received fraction is exp(−2αL).
    let alpha = 5.0;
    let l = 0.05;
    let recv = received_signal_fraction(
        1.5e6, 1.5e6, alpha, l, 0.0, 0.5e6, 2e-6, 1481.0, 998.0, 1e-3, 101_325.0, 1.4,
    );
    let expected = (-2.0 * alpha * l).exp(); // T_iface = 1 at matched impedance
    assert!(
        (recv - expected).abs() < 1e-12,
        "two-way: {recv} vs {expected}"
    );
}

#[test]
fn interface_pressure_transmission_physics() {
    // Pressure transmission T = 2z2/(z1+z2): into a HIGHER impedance the
    // pressure amplitude rises (T>1, intensity still conserved); into a LOWER
    // impedance it drops (T<1). Matched impedance transmits unchanged.
    let into_higher = pressure_transmission_coefficient(1.38e6, 1.65e6); // fat→liver
    let into_lower = pressure_transmission_coefficient(1.65e6, 1.38e6); // liver→fat
    assert!(
        into_higher > 1.0,
        "into higher Z, pressure rises: {into_higher}"
    );
    assert!(
        into_lower < 1.0 && into_lower > 0.0,
        "into lower Z, pressure drops: {into_lower}"
    );
    assert!((pressure_transmission_coefficient(1.5e6, 1.5e6) - 1.0).abs() < 1e-12);
    // Strong mismatch toward a much lower impedance (gas) collapses transmission.
    let into_gas = pressure_transmission_coefficient(1.5e6, 4.1e2);
    assert!(
        into_gas < 0.01,
        "huge impedance drop nearly blocks transmission: {into_gas}"
    );
}

#[test]
fn pulses_for_lesion_radius_inverts_forward_model() {
    use crate::analytical::cavitation::histotripsy_lesion_radius_m;
    let r0 = 3e-6;
    let p0 = 101_325.0;
    let sigma_y = 2.0e3;
    let icd_per_pulse = 50.0;
    let target = 1.0e-3; // 1 mm lesion
    let n = histotripsy_pulses_for_lesion_radius(target, r0, p0, sigma_y, icd_per_pulse);
    assert!(n > 0.0);
    // Round-trip: feeding N·icd_per_pulse back into the forward model recovers
    // the target radius.
    let r_back = histotripsy_lesion_radius_m(n * icd_per_pulse, r0, p0, sigma_y);
    assert!(
        (r_back - target).abs() / target < 1e-9,
        "inverse must recover target radius: {r_back} vs {target}"
    );
    // A tighter safety margin (smaller target) needs fewer pulses.
    let n_tight = histotripsy_pulses_for_lesion_radius(0.5e-3, r0, p0, sigma_y, icd_per_pulse);
    assert!(n_tight < n);
}
