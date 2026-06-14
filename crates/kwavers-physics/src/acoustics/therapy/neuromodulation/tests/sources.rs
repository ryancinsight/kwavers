//! Auto-organised from the original single-file `tests.rs` (split by concern).

use super::*;

#[test]
fn capacitance_waveform_matches_analytic_derivative() {
    let m = CapacitanceModulation::new(1.0, 0.3, 0.5);
    assert!(m.is_valid());
    // C_m(0) = C_m0; rate(0) = C_m0·ε·ω (cos(0) = 1).
    assert!((m.capacitance(0.0) - 1.0).abs() < 1e-12);
    assert!((m.capacitance_rate(0.0) - 0.3 * m.omega_rad_ms).abs() < 1e-9);
    // Central finite difference of C_m matches the analytic rate.
    let t = 3.7e-4_f64; // ms (well within one 0.5 MHz period of 2e-3 ms)
    let dh = 1e-9;
    let fd = (m.capacitance(t + dh) - m.capacitance(t - dh)) / (2.0 * dh);
    assert!(
        (fd - m.capacitance_rate(t)).abs() < 1e-3 * m.capacitance_rate(0.0).abs(),
        "FD {fd} vs analytic {}",
        m.capacitance_rate(t)
    );
}

#[test]
fn modulation_depth_scales_linearly_and_clamps() {
    let r = 10.0e-6; // 10 µm
    let ka = BILAYER_AREA_MODULUS_N_M;
    // ε = p·R/(2·K_A). At 10 kPa: 1e4·1e-5/(2·0.24) ≈ 0.2083.
    let eps = modulation_depth_from_pressure(1.0e4, r, ka);
    assert!((eps - 1.0e4 * r / (2.0 * ka)).abs() < 1e-12);
    assert!((eps - 0.2083).abs() < 1e-3, "ε = {eps}");
    // Linear scaling below saturation.
    let eps2 = modulation_depth_from_pressure(2.0e4, r, ka);
    assert!(
        (eps2 - 2.0 * eps).abs() < 1e-9,
        "not linear: {eps} → {eps2}"
    );
    // Clamp at large pressure; reject non-physical inputs.
    assert!((modulation_depth_from_pressure(1.0e6, r, ka) - 0.99).abs() < 1e-12);
    assert_eq!(modulation_depth_from_pressure(-1.0, r, ka), 0.0);
    assert_eq!(modulation_depth_from_pressure(1.0e4, 0.0, ka), 0.0);
    assert_eq!(modulation_depth_from_pressure(1.0e4, r, 0.0), 0.0);
}

#[test]
fn bls_capacitance_geometry_matches_eq8() {
    use super::super::bls::{bls_capacitance, LEAFLET_GAP_M, SONOPHORE_RADIUS_M};
    let (cm0, a, d) = (1.0, SONOPHORE_RADIUS_M, LEAFLET_GAP_M);
    // Z = 0 ⇒ rest capacitance.
    assert!((bls_capacitance(0.0, cm0, a, d) - cm0).abs() < 1e-9);
    // Independent hand evaluation of Eq. 8 at Z = 2 nm (a=32 nm, Δ=1.26 nm):
    // factor = Δ/a² = 1.230e6; bracket ≈ 3.656e-7 m ⇒ C_m ≈ 0.4498·C_m0.
    let c2 = bls_capacitance(2.0e-9, cm0, a, d);
    assert!((c2 - 0.4498).abs() < 5e-3, "C_m(2nm) = {c2}");
    // Capacitance falls monotonically as the cavity widens (gap increases).
    let seq: Vec<f64> = [0.0, 0.5e-9, 1.0e-9, 2.0e-9, 4.0e-9]
        .iter()
        .map(|&z| bls_capacitance(z, cm0, a, d))
        .collect();
    assert!(
        seq.windows(2).all(|w| w[1] < w[0]),
        "C_m(Z) not monotone decreasing: {seq:?}"
    );
    // Just above the series threshold, the full Eq. 8 branch agrees with the
    // analytic small-Z series C_m ≈ C_m0·(1 − Z/Δ) (no jump discontinuity).
    let z_near = 2.0e-4 * d;
    assert!(
        (bls_capacitance(z_near, cm0, a, d) - cm0 * (1.0 - z_near / d)).abs() < 1e-6,
        "series/full mismatch near Z=0"
    );
}

#[test]
fn bls_capacitance_rises_under_compression() {
    // Regression guard: the parallel-plate capacitance must *rise* above C_m0 as
    // the cavity compresses (Z<0 narrows the inter-leaflet gap) — Eq. 8 applies
    // for both signs of Z (the earlier `z.max(0.0)` clamp wrongly returned C_m0).
    use super::super::bls::{bls_capacitance, LEAFLET_GAP_M, SONOPHORE_RADIUS_M};
    let (cm0, a, d) = (1.0, SONOPHORE_RADIUS_M, LEAFLET_GAP_M);
    // Compression (Z<0) ⇒ C_m > C_m0, growing as the gap narrows toward the wall.
    let seq: Vec<f64> = [0.0, -0.05e-9, -0.1e-9, -0.2e-9, -0.3e-9]
        .iter()
        .map(|&z| bls_capacitance(z, cm0, a, d))
        .collect();
    assert!((seq[0] - cm0).abs() < 1e-9, "C_m(0) = {}", seq[0]);
    assert!(
        seq.windows(2).all(|w| w[1] > w[0]),
        "C_m(Z) not monotone increasing under compression: {seq:?}"
    );
    // Small-Z series symmetry: C_m(−z) ≈ C_m0·(1 + z/Δ) = 2·C_m0 − C_m(+z).
    let zs = 1.0e-4 * d;
    assert!(
        (bls_capacitance(-zs, cm0, a, d) - cm0 * (1.0 + zs / d)).abs() < 1e-6,
        "compression series mismatch"
    );
}

#[test]
fn bls_pressure_rest_balance_and_gap() {
    use super::super::bls::pressures as bp;
    // Resting charge density [C/m²] = C_m0 [F/m²] · V_rest [V].
    let qm0 = 1.0e-2 * (-71.9e-3);
    let delta = bp::rest_gap(qm0);
    // The solved rest gap reproduces Plaksin's Δ ≈ 1.26 nm from the charge balance.
    assert!(
        (delta - 1.26e-9).abs() < 0.05e-9,
        "rest gap {delta:e} m (expected ≈ 1.26 nm)"
    );
    // At Z = 0 with the solved gap the static total pressure vanishes (gas at P0).
    let ng0 = bp::initial_gas_mol(delta);
    let p0 = bp::static_total_pressure(0.0, ng0, qm0, 0.0, delta);
    assert!(p0.abs() < 1.0, "rest P_tot(0) = {p0} Pa (expected ≈ 0)");
}

#[test]
fn bls_quasistatic_deflection_is_pressure_driven_and_rectified() {
    use super::super::bls::pressures as bp;
    let qm0 = 1.0e-2 * (-71.9e-3);
    let delta = bp::rest_gap(qm0);
    // Compression (pac > 0) cannot deflect the cavity below flat ⇒ Z = 0.
    assert_eq!(bp::quasistatic_deflection(300.0e3, qm0, delta), 0.0);
    // Rarefaction (pac < 0) expands the cavity; deflection grows with amplitude.
    let z100 = bp::quasistatic_deflection(-100.0e3, qm0, delta);
    let z300 = bp::quasistatic_deflection(-300.0e3, qm0, delta);
    let z500 = bp::quasistatic_deflection(-500.0e3, qm0, delta);
    assert!(
        z100 > 0.0 && z300 > z100 && z500 > z300,
        "not monotone: {z100} {z300} {z500}"
    );
    // Order-of-magnitude agreement with Plaksin Fig. 1 (≈ 12 nm, ≈ 15 mN/m at
    // 500 kPa / 0.5 MHz; quasi-static under-deflects vs the resonant dynamics).
    assert!((5.0e-9..20.0e-9).contains(&z500), "Z(500 kPa) = {z500:e} m");
    let tension = bp::elastic_tension(z500);
    assert!(
        (5.0e-3..40.0e-3).contains(&tension),
        "tension = {tension} N/m"
    );
}

#[test]
fn bls_dynamic_deflection_inertially_amplified_near_plaksin_fig1() {
    // Value-semantic validation of the transient (inertial) leaflet ODE against
    // Plaksin et al. (2014, Phys. Rev. X 4, 011004) Fig. 1 at 500 kPa / 0.5 MHz
    // on the cortical RS membrane (rest −71.9 mV).
    //
    // Two physically-meaningful checks:
    //  (1) the inertial dynamics **amplify** the deflection well above the
    //      quasi-static value at the same peak rarefaction pressure (the whole
    //      reason the transient ODE exists); and
    //  (2) the steady-cycle peak lands in a band centred on the model's genuine
    //      value (≈10.5 nm) and reaching the published ≈12 nm reference. The
    //      reduced single-leaflet model with the tabulated `P_M(Z)` and
    //      quasi-static intra-cycle charge sits ≈13 % below the full two-leaflet
    //      Plaksin result — empirical agreement at the reduced-model tier, not a
    //      free parameter (tuning the leaflet damping to hit 12 nm exactly would
    //      be fitting-to-target; the ODE coefficients are fixed by the physics).
    use super::super::bls::{pressures as bp, BilayerSonophoreDynamic};
    let dyn_src = BilayerSonophoreDynamic::new(1.0, 0.5, 500.0e3, -71.9);
    let peak_nm = dyn_src.peak_deflection_m() * 1.0e9;

    let qm0 = 1.0e-2 * (-71.9e-3);
    let delta = bp::rest_gap(qm0);
    let quasistatic_nm = bp::quasistatic_deflection(-500.0e3, qm0, delta) * 1.0e9;

    assert!(
        peak_nm > quasistatic_nm,
        "inertial dynamics must deflect at least as far as the quasi-static balance: \
         transient {peak_nm:.2} nm vs quasi-static {quasistatic_nm:.2} nm"
    );
    assert!(
        (9.8..11.5).contains(&peak_nm),
        "transient peak deflection {peak_nm:.2} nm must match the documented \
         reduced-model value (≈10.5 nm, ≈13 % below the published Plaksin Fig. 1 ≈12 nm)"
    );
}

#[test]
fn phase_cycle_interpolates_and_differentiates() {
    // PhaseCycle is the SSOT for precomputed periodic capacitance sources: it
    // interpolates C_m(t) by phase and derives dC_m/dt by central difference.
    use super::super::intramembrane_cavitation::{CapacitanceSource, PhaseCycle};
    use std::f64::consts::PI;
    let n = 720;
    let cm0 = 1.0;
    let depth = 0.2;
    let freq_mhz = 0.5;
    let omega = 2.0 * PI * 1.0e3 * freq_mhz; // rad/ms
                                             // A pure sinusoid C_m(phase) = cm0(1 + ε·sin(phase)); index 0 = phase 0.
    let cm_cycle: Vec<f64> = (0..n)
        .map(|i| cm0 * (1.0 + depth * ((i as f64 / n as f64) * 2.0 * PI).sin()))
        .collect();
    let pc = PhaseCycle::new(cm0, omega, cm_cycle);
    assert!(pc.is_source_valid());
    assert!((pc.baseline_capacitance() - cm0).abs() < 1e-12);
    // At t=0, C_m = cm0 (sin 0); analytic dC_m/dt = cm0·ε·ω.
    assert!((pc.capacitance(0.0) - cm0).abs() < 1e-3);
    assert!(
        (pc.capacitance_rate(0.0) - cm0 * depth * omega).abs() < 1e-2 * (cm0 * depth * omega),
        "rate(0) = {} vs {}",
        pc.capacitance_rate(0.0),
        cm0 * depth * omega
    );
    // Periodicity: C_m(t + period) = C_m(t).
    let period = 2.0 * PI / omega;
    let t = 0.37 * period;
    assert!((pc.capacitance(t) - pc.capacitance(t + period)).abs() < 1e-9);
}

#[test]
fn bls_dynamic_deflection_strength_curve_vs_plaksin_fig1() {
    // ADR 023 validation harness: the transient leaflet deflection as a function
    // of acoustic amplitude at 0.5 MHz (cortical RS rest), the strength curve of
    // Plaksin et al. (2014) Fig. 1. Pins the model's genuine `Z_peak(P_ac)` at
    // three amplitudes and checks the qualitative reference behaviour.
    //
    // | P_ac    | model Z_peak | Plaksin Fig. 1 (≈) |
    // |---------|--------------|--------------------|
    // | 100 kPa | 5.37 nm      | —                  |
    // | 300 kPa | 8.59 nm      | —                  |
    // | 500 kPa | 10.45 nm     | ≈12 nm             |
    //
    // The reduced single-`Z` model is overdamped/below-resonance: the deflection
    // reaches steady amplitude within one carrier cycle (no resonant build-up) and
    // sits ≈13 % below Fig. 1 at 500 kPa. Closing the gap to <5 % across this whole
    // curve is the [major] reference-calibration tracked by ADR 023; it needs the
    // digitised Fig. 1 curve + a PySONIC differential audit and must NOT be reached
    // by tuning the leaflet damping (fitting-to-target).
    use super::super::bls::{pressures as bp, BilayerSonophoreDynamic};
    let qm0 = 1.0e-2 * (-71.9e-3);
    let delta = bp::rest_gap(qm0);

    let peaks_nm: Vec<f64> = [100.0e3, 300.0e3, 500.0e3]
        .iter()
        .map(|&p| BilayerSonophoreDynamic::new(1.0, 0.5, p, -71.9).peak_deflection_m() * 1.0e9)
        .collect();

    // Monotone increasing with amplitude (strength curve).
    assert!(
        peaks_nm.windows(2).all(|w| w[1] > w[0]),
        "Z_peak(P) must increase with amplitude: {peaks_nm:?} nm"
    );
    // Each transient peak is at least the quasi-static deflection at that pressure
    // (inertial dynamics never deflect *less* than the static balance).
    for (&p, &zd) in [100.0e3, 300.0e3, 500.0e3].iter().zip(&peaks_nm) {
        let zq = bp::quasistatic_deflection(-p, qm0, delta) * 1.0e9;
        assert!(
            zd >= zq - 1.0e-3,
            "transient {zd:.2} nm below quasi-static {zq:.2} nm at {:.0} kPa",
            p / 1.0e3
        );
    }
    // Pin the genuine model values (regression guard on the documented numbers).
    let expected = [5.37, 8.59, 10.45];
    for (i, (&got, &exp)) in peaks_nm.iter().zip(&expected).enumerate() {
        assert!(
            (got - exp).abs() < 0.4,
            "amplitude {i}: Z_peak {got:.2} nm drifted from documented {exp:.2} nm"
        );
    }
}
