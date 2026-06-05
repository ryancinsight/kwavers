use super::*;

// ─── emission ────────────────────────────────────────────────────────────────

#[test]
fn emission_constant_radius_is_silent() {
    // Ṙ = 0 ⇒ R̈ = 0 ⇒ p_sc = 0 at every sample.
    let r = vec![2e-6_f64; 8];
    let rdot = vec![0.0_f64; 8];
    let p = bubble_acoustic_emission_pressure(&r, &rdot, 1e-8, 998.0, 0.05);
    assert_eq!(p.len(), 8);
    assert!(
        p.iter().all(|&x| x == 0.0),
        "silent bubble must emit nothing"
    );
}

#[test]
fn emission_matches_closed_form() {
    // Exact check of p = ρR/r_obs·(2Ṙ² + R·R̈) with hand-chosen arrays.
    // dt=1, ρ=1, r_obs=1; rdot ramps by 1 ⇒ central-diff R̈ = 1 (interior),
    // forward/backward R̈ = 1 at the ends.
    let r = vec![2.0_f64; 5];
    let rdot = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let p = bubble_acoustic_emission_pressure(&r, &rdot, 1.0, 1.0, 1.0);
    // i=0: R̈=1, Ṙ=0 → 1·2·(0 + 2·1) = 4
    assert!((p[0] - 4.0).abs() < 1e-12, "p[0]={}", p[0]);
    // i=2: R̈=1, Ṙ=2 → 1·2·(2·4 + 2·1) = 20
    assert!((p[2] - 20.0).abs() < 1e-12, "p[2]={}", p[2]);
    // i=4 (backward diff R̈=(4-3)/1=1), Ṙ=4 → 2·(2·16 + 2) = 68
    assert!((p[4] - 68.0).abs() < 1e-12, "p[4]={}", p[4]);
}

#[test]
fn emission_rejects_bad_input() {
    assert!(bubble_acoustic_emission_pressure(&[1.0], &[0.0], 1.0, 1.0, 1.0).is_empty());
    assert!(bubble_acoustic_emission_pressure(&[1.0, 2.0], &[0.0], 1.0, 1.0, 1.0).is_empty());
    assert!(bubble_acoustic_emission_pressure(&[1.0, 2.0], &[0.0, 1.0], 0.0, 1.0, 1.0).is_empty());
}

// ─── spectral band decomposition ─────────────────────────────────────────────

fn line_spectrum() -> (Vec<f64>, Vec<f64>, f64) {
    // f0 = 1 MHz, Δf = 0.1 MHz, bins 0..2.0 MHz.
    let f0 = 1.0e6;
    let freqs: Vec<f64> = (0..=20).map(|k| k as f64 * 0.1e6).collect();
    let mut psd = vec![0.0_f64; freqs.len()];
    // subharmonic 0.5 MHz (idx 5), fundamental 1.0 MHz (idx 10),
    // ultraharmonic 1.5 MHz (idx 15), harmonic 2.0 MHz (idx 20),
    // broadband 0.7 MHz (idx 7).
    psd[5] = 3.0; // sub
    psd[10] = 10.0; // fundamental
    psd[20] = 4.0; // 2f0 harmonic
    psd[15] = 2.0; // ultra
    psd[7] = 1.0; // broadband
    (freqs, psd, f0)
}

#[test]
fn bands_classify_lines_exactly() {
    let (freqs, psd, f0) = line_spectrum();
    let df = 0.1e6;
    let b = decompose_emission_spectrum(&freqs, &psd, f0, 0.04, 0.0);
    assert!(
        (b.subharmonic - 3.0 * df).abs() < 1e-3,
        "sub={}",
        b.subharmonic
    );
    assert!(
        (b.fundamental - (10.0 + 4.0) * df).abs() < 1e-3,
        "fund={}",
        b.fundamental
    );
    assert!(
        (b.ultraharmonic - 2.0 * df).abs() < 1e-3,
        "ultra={}",
        b.ultraharmonic
    );
    assert!(
        (b.broadband - 1.0 * df).abs() < 1e-3,
        "broad={}",
        b.broadband
    );
}

#[test]
fn bands_are_additive() {
    let (freqs, psd, f0) = line_spectrum();
    let b = decompose_emission_spectrum(&freqs, &psd, f0, 0.04, 0.0);
    let total_above_floor: f64 = psd.iter().sum::<f64>() * (freqs[1] - freqs[0]);
    let sum = b.fundamental + b.subharmonic + b.ultraharmonic + b.broadband;
    assert!(
        (sum - total_above_floor).abs() < 1e-6,
        "bands sum {sum} ≠ total {total_above_floor}"
    );
}

#[test]
fn bands_subtract_noise_floor() {
    let (freqs, psd, f0) = line_spectrum();
    // Floor at 10.5 exceeds every bin ⇒ all bands vanish.
    let b = decompose_emission_spectrum(&freqs, &psd, f0, 0.04, 10.5);
    assert_eq!(b.subharmonic, 0.0);
    assert_eq!(b.fundamental, 0.0);
    assert_eq!(b.ultraharmonic, 0.0);
    assert_eq!(b.broadband, 0.0);
}

#[test]
fn bands_stable_and_inertial_helpers() {
    let (freqs, psd, f0) = line_spectrum();
    let df = 0.1e6;
    let b = decompose_emission_spectrum(&freqs, &psd, f0, 0.04, 0.0);
    assert!((b.stable_emission() - (3.0 + 2.0) * df).abs() < 1e-3);
    assert!((b.inertial_emission() - 1.0 * df).abs() < 1e-3);
}

#[test]
fn bands_reject_bad_input() {
    let z = decompose_emission_spectrum(&[], &[], 1e6, 0.05, 0.0);
    assert_eq!(z.fundamental, 0.0);
    assert_eq!(z.broadband, 0.0);
}

// ─── dose accumulation ───────────────────────────────────────────────────────

#[test]
fn cumulative_dose_of_constant_power_is_linear() {
    // Constant P=2 over 5 samples at dt=0.5: trapezoid is exact ⇒ D[m]=P·m·dt.
    let d = cumulative_cavitation_dose(&[2.0; 5], 0.5);
    assert_eq!(d[0], 0.0);
    for (m, &dm) in d.iter().enumerate() {
        let expected = 2.0 * m as f64 * 0.5;
        assert!((dm - expected).abs() < 1e-12, "D[{m}]={dm} ≠ {expected}");
    }
}

#[test]
fn cumulative_dose_clamps_negative_power() {
    let d = cumulative_cavitation_dose(&[0.0, -5.0, -5.0], 1.0);
    assert!(
        d.iter().all(|&x| x == 0.0),
        "negative power must not accumulate"
    );
}

#[test]
fn cumulative_dose_monotone_nondecreasing() {
    let d = cumulative_cavitation_dose(&[1.0, 3.0, 0.0, 4.0, 2.0], 0.1);
    for k in 1..d.len() {
        assert!(d[k] >= d[k - 1], "dose must not decrease at {k}");
    }
}

// ─── controller ──────────────────────────────────────────────────────────────

#[test]
fn controller_backs_off_on_inertial() {
    // Inertial emission over the limit ⇒ multiplicative back-off, safety first
    // even though stable is also below target.
    let next = cavitation_controller_pressure(100.0, 0.0, 5.0, 10.0, 1.0, 0.1, 10.0, 1000.0);
    assert!((next - 90.0).abs() < 1e-9, "expected 90, got {next}");
}

#[test]
fn controller_recruits_when_under_target_and_safe() {
    let next = cavitation_controller_pressure(100.0, 2.0, 0.5, 10.0, 1.0, 0.1, 10.0, 1000.0);
    assert!((next - 110.0).abs() < 1e-9, "expected 110, got {next}");
}

#[test]
fn controller_holds_inside_window() {
    let next = cavitation_controller_pressure(100.0, 12.0, 0.5, 10.0, 1.0, 0.1, 10.0, 1000.0);
    assert!(
        (next - 100.0).abs() < 1e-9,
        "expected hold at 100, got {next}"
    );
}

#[test]
fn controller_respects_pressure_clamp() {
    // Recruiting from 995 with +10% would exceed p_max=1000 ⇒ clamps.
    let next = cavitation_controller_pressure(995.0, 0.0, 0.0, 10.0, 1.0, 0.1, 10.0, 1000.0);
    assert!(
        (next - 1000.0).abs() < 1e-9,
        "expected clamp at 1000, got {next}"
    );
}

// ─── receiver / volume integration ───────────────────────────────────────────

#[test]
fn receiver_array_incoherent_power_sum() {
    // 2 channels × 3 bins, row-major.
    let ch = vec![1.0, 2.0, 3.0, /*ch1*/ 4.0, 5.0, 6.0];
    let s = integrate_receiver_array_psd(&ch, 2, 3);
    assert_eq!(s, vec![5.0, 7.0, 9.0]);
}

#[test]
fn receiver_array_rejects_shape_mismatch() {
    assert!(integrate_receiver_array_psd(&[1.0, 2.0, 3.0], 2, 3).is_empty());
    assert!(integrate_receiver_array_psd(&[], 0, 3).is_empty());
}

#[test]
fn volume_integration_counts_masked_voxels() {
    // 4 voxels, mask selects voxels 0 and 2; source all = 2; dv = 0.5.
    let source = vec![2.0, 2.0, 2.0, 2.0];
    let mask = vec![1.0, 0.0, 1.0, 0.0];
    let e = emission_energy_in_volume(&source, &mask, 0.5);
    assert!((e - (2.0 + 2.0) * 0.5).abs() < 1e-12, "E={e}");
}

#[test]
fn volume_integration_clamps_negative_source() {
    let source = vec![-3.0, 5.0];
    let mask = vec![1.0, 1.0];
    let e = emission_energy_in_volume(&source, &mask, 1.0);
    assert!((e - 5.0).abs() < 1e-12, "negative source must clamp: E={e}");
}

#[test]
fn volume_integration_rejects_bad_input() {
    assert_eq!(emission_energy_in_volume(&[1.0], &[1.0, 2.0], 1.0), 0.0);
    assert_eq!(emission_energy_in_volume(&[1.0], &[1.0], 0.0), 0.0);
}

// ─── true adaptive Keller–Miksis emission simulation ─────────────────────────

fn base_drive(drive_amp_pa: f64, r0_m: f64) -> BubbleDriveConfig {
    BubbleDriveConfig {
        r0_m,
        p0_pa: 101_325.0,
        rho: 998.0,
        c_liquid: 1481.0,
        mu: 1.0e-3,
        sigma: 0.0725,
        pv: 2330.0,
        gamma: 1.4,
        drive_freq_hz: 1.0e6,
        drive_amp_pa,
        n_cycles: 8.0,
        n_out: 1024,
        r_obs_m: 5.0e-2,
        thermal_effects: false,
    }
}

#[test]
fn simulate_low_drive_is_finite_and_near_equilibrium() {
    // Gentle drive: bubble oscillates mildly, stays finite, no violent collapse.
    let tr = simulate_bubble_emission(&base_drive(20.0e3, 2.0e-6));
    assert!(tr.converged, "low-drive sim should converge");
    assert_eq!(tr.time.len(), tr.emission.len());
    assert!(tr.emission.iter().all(|x| x.is_finite()));
    assert!(tr.radius.iter().all(|&r| r > 0.0 && r.is_finite()));
    // Mild drive ⇒ modest compression.
    assert!(
        tr.max_compression < 5.0,
        "low drive over-compressed: {}",
        tr.max_compression
    );
}

#[test]
fn simulate_survives_strong_inertial_drive() {
    // Strong drive that blows up a fixed-step RK4: the adaptive solver must
    // keep R(t) finite and positive (it may report non-convergence at the
    // stiffest collapse, but must not return NaN/Inf samples).
    let tr = simulate_bubble_emission(&base_drive(800.0e3, 2.0e-6));
    assert!(
        tr.time.len() >= 2,
        "should produce at least the seed samples"
    );
    assert!(
        tr.radius.iter().all(|&r| r.is_finite() && r > 0.0),
        "radii must stay finite/positive under adaptive stepping"
    );
    assert!(
        tr.emission.iter().all(|x| x.is_finite()),
        "emission must stay finite"
    );
}

#[test]
fn simulate_stronger_drive_increases_compression_and_emission() {
    let weak = simulate_bubble_emission(&base_drive(40.0e3, 2.0e-6));
    let strong = simulate_bubble_emission(&base_drive(150.0e3, 2.0e-6));
    let energy = |t: &BubbleEmissionTrace| t.emission.iter().map(|x| x * x).sum::<f64>();
    assert!(
        strong.max_compression >= weak.max_compression,
        "stronger drive should compress at least as much"
    );
    assert!(
        energy(&strong) > energy(&weak),
        "stronger drive should radiate more emission energy"
    );
}

#[test]
fn simulate_rejects_bad_input() {
    let bad = simulate_bubble_emission(&base_drive(50e3, -1.0e-6));
    assert!(!bad.converged);
}

#[test]
fn thermal_effects_no_runaway_expansion() {
    // Regression: with full gas thermodynamics + mass transfer enabled, a 3 µm
    // bubble at 800 kPa previously blew up to ~mm scale (vapor mass-transfer
    // overshoot). The equilibrium clamp in update_mass_transfer keeps the radius
    // physically bounded (a real inertial expansion is at most ~10–20× R₀).
    let mut cfg = base_drive(800.0e3, 3.0e-6);
    cfg.thermal_effects = true;
    cfg.n_cycles = 8.0;
    let tr = simulate_bubble_emission(&cfg);
    let rmax = tr.radius.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        rmax.is_finite() && rmax < 100.0 * cfg.r0_m,
        "thermal mass-transfer runaway: Rmax={rmax:.3e} m (R0={:.1e})",
        cfg.r0_m
    );
}

fn base_shell(drive_amp_pa: f64, r0_m: f64) -> ShellDriveConfig {
    ShellDriveConfig {
        r0_m,
        p0_pa: 101_325.0,
        rho: 998.0,
        c_liquid: 1481.0,
        mu: 1.0e-3,
        gamma: 1.4,
        drive_freq_hz: 1.0e6,
        drive_amp_pa,
        n_cycles: 10.0,
        steps_per_cycle: 2000,
        n_out: 4096,
        r_obs_m: 5.0e-2,
        chi: 0.5,
        shell_viscosity: 0.5,
        shell_thickness: 3.0e-9,
        sigma_initial: 0.04,
    }
}

#[test]
#[ignore]
fn probe_coated_subharmonic() {
    let f0 = 1.0e6;
    let frac = |emit: &[f64], dt: f64| -> (f64, f64, f64) {
        let (fr, psd) = hann_windowed_power_spectrum(emit, dt, emit.len());
        let b = decompose_emission_spectrum(&fr, &psd, f0, 0.1, 0.0);
        let tot = b.fundamental + b.subharmonic + b.ultraharmonic + b.broadband + 1e-30;
        (
            b.subharmonic / tot,
            b.ultraharmonic / tot,
            b.broadband / tot,
        )
    };
    for r0u in [1.0, 2.0, 3.0, 4.0] {
        for pa in [80e3, 150e3, 300e3, 500e3] {
            for chi in [0.25, 0.5, 1.0] {
                let mut c = base_shell(pa, r0u * 1e-6);
                c.chi = chi;
                let tr = simulate_coated_bubble_emission(&c);
                let dt = tr.time[1] - tr.time[0];
                let (s, u, b) = frac(&tr.emission, dt);
                println!(
                    "r0={r0u}um pa={:.0}kPa chi={chi} -> sub={:.1}% ultra={:.1}% broad={:.1}% maxC={:.1}",
                    pa / 1e3, s * 100.0, u * 100.0, b * 100.0, tr.max_compression
                );
            }
        }
    }
}

#[test]
fn coated_bubble_emission_is_finite() {
    let tr = simulate_coated_bubble_emission(&base_shell(100.0e3, 2.0e-6));
    assert!(tr.converged, "shell sim should converge at clinical drive");
    assert_eq!(tr.time.len(), tr.emission.len());
    assert!(tr.emission.iter().all(|x| x.is_finite()));
    assert!(tr.radius.iter().all(|&r| r > 0.0 && r.is_finite()));
}

#[test]
fn coated_bubble_emits_subharmonic_above_free_bubble() {
    // The shell buckling/rupture nonlinearity produces a subharmonic (f0/2) that
    // a free bubble at the same drive does not. Compare the f0/2-band energy
    // fraction of the coated vs free bubble emission spectra.
    let f0 = 1.0e6;
    let band_sub_frac = |emit: &[f64], dt: f64| -> f64 {
        if emit.len() < 16 {
            return 0.0;
        }
        let (fr, psd) = hann_windowed_power_spectrum(emit, dt, emit.len());
        let b = decompose_emission_spectrum(&fr, &psd, f0, 0.1, 0.0);
        b.subharmonic / (b.fundamental + b.subharmonic + b.ultraharmonic + b.broadband + 1e-30)
    };
    // Coated 2 µm bubble in the subharmonic regime (soft shell, 300 kPa): the
    // Marmottant buckling nonlinearity period-doubles where a free bubble at the
    // same drive does not.
    let mut sc = base_shell(300.0e3, 2.0e-6);
    sc.chi = 0.25;
    let shell = simulate_coated_bubble_emission(&sc);
    let dt_s =
        shell.time.get(1).copied().unwrap_or(1.0) - shell.time.first().copied().unwrap_or(0.0);
    let free = simulate_bubble_emission(&base_drive(300.0e3, 2.0e-6));
    let dt_f = free.time.get(1).copied().unwrap_or(1.0) - free.time.first().copied().unwrap_or(0.0);
    let sub_shell = band_sub_frac(&shell.emission, dt_s.max(1e-12));
    let sub_free = band_sub_frac(&free.emission, dt_f.max(1e-12));
    assert!(
        sub_shell >= sub_free,
        "coated bubble subharmonic fraction {sub_shell} should be >= free {sub_free}"
    );
}

// ─── ensemble superposition ──────────────────────────────────────────────────

#[test]
fn ensemble_sums_delayed_gained_series() {
    // 2 bubbles × 3 samples; bubble 0 at delay 0 gain 1, bubble 1 at delay 2 gain 2.
    let emissions = vec![1.0, 2.0, 3.0, /*b1*/ 1.0, 1.0, 1.0];
    let out = ensemble_emission_superposition(&emissions, 2, 3, &[0, 2], &[1.0, 2.0], 6);
    // b0 → [1,2,3,0,0,0]; b1 (×2, +2) → [0,0,2,2,2,0]; sum:
    assert_eq!(out, vec![1.0, 2.0, 5.0, 2.0, 2.0, 0.0]);
}

#[test]
fn ensemble_truncates_overrun_tail() {
    // delay 2 + len 3 = 5 > out_len 4 → last sample truncated, no panic.
    let emissions = vec![1.0, 1.0, 1.0];
    let out = ensemble_emission_superposition(&emissions, 1, 3, &[2], &[1.0], 4);
    assert_eq!(out, vec![0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn ensemble_rejects_bad_shape() {
    assert!(ensemble_emission_superposition(&[1.0, 2.0], 2, 3, &[0, 0], &[1.0, 1.0], 5).is_empty());
    assert!(ensemble_emission_superposition(&[1.0, 2.0, 3.0], 1, 3, &[0], &[1.0], 0).is_empty());
}

// ─── windowed spectrum ───────────────────────────────────────────────────────

#[test]
fn hann_spectrum_peaks_at_tone_frequency() {
    // Pure tone at f_tone resolves to its bin; Hann concentrates energy there.
    let dt = 1e-7_f64; // 10 MHz sample rate
    let n = 1024usize;
    let f_tone = 1.0e6;
    let sig: Vec<f64> = (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * f_tone * i as f64 * dt).sin())
        .collect();
    let (freqs, psd) = hann_windowed_power_spectrum(&sig, dt, n);
    let kmax = psd
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap();
    assert!(
        (freqs[kmax] - f_tone).abs() < freqs[1] - freqs[0] + 1.0,
        "peak at {} Hz, expected {f_tone} Hz",
        freqs[kmax]
    );
    // Off-peak leakage is far below the peak (Hann sidelobes).
    let peak = psd[kmax];
    let far = psd[(kmax + 50).min(psd.len() - 1)];
    assert!(far < 1e-3 * peak, "leakage {far} not << peak {peak}");
}

#[test]
fn hann_spectrum_rejects_bad_input() {
    assert!(hann_windowed_power_spectrum(&[1.0, 2.0], 0.0, 4)
        .0
        .is_empty());
    assert!(hann_windowed_power_spectrum(&[1.0, 2.0], 1.0, 1)
        .0
        .is_empty());
}

// ─── end-to-end: KM-driven emission → bands → dose ───────────────────────────

#[test]
fn pipeline_drive_raises_emission_and_harmonic_content() {
    use crate::analytical::cavitation::{bubble_power_spectrum, keller_miksis_rk4};
    // Drive a 2 µm air bubble in water near resonance: low vs high pressure.
    // A single deterministic bubble radiates a *line* spectrum (harmonics, and
    // sub/ultraharmonics once period-doubled), so the physically robust claim
    // is that both total radiated emission and nonlinear harmonic content grow
    // with drive amplitude — not that broadband appears (broadband needs an
    // asynchronous bubble population, modelled at the receiver-integration
    // layer, not in one KM trajectory).
    let r0 = 2e-6;
    let f0 = 1.0e6;
    let dt = 2e-9;
    let n = 4096usize;
    let t: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
    let run = |p_ac: f64| -> super::CavitationBandEnergies {
        let (r, rdot) = keller_miksis_rk4(
            r0, 0.0, p_ac, f0, &t, 101_325.0, 998.0, 0.0725, 1.0e-3, 1.4, 2330.0, 1481.0,
        );
        let emit = bubble_acoustic_emission_pressure(&r, &rdot, dt, 998.0, 5e-2);
        let (freqs, psd) = bubble_power_spectrum(&emit, dt, n);
        decompose_emission_spectrum(&freqs, &psd, f0, 0.04, 0.0)
    };
    let total = |b: &super::CavitationBandEnergies| {
        b.fundamental + b.subharmonic + b.ultraharmonic + b.broadband
    };
    let low = run(20.0e3); // gentle, near-linear oscillation
    let high = run(80.0e3); // stronger, more nonlinear (KM-stable at dt=2ns)
    for b in [&low, &high] {
        assert!(b.fundamental.is_finite() && b.broadband.is_finite());
    }
    assert!(
        total(&high) > total(&low),
        "stronger drive must radiate more: low={}, high={}",
        total(&low),
        total(&high)
    );
    assert!(
        high.fundamental > low.fundamental,
        "harmonic emission must grow with drive: low={}, high={}",
        low.fundamental,
        high.fundamental
    );
}
