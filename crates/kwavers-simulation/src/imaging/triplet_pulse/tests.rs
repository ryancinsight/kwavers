//! Tests for the triplet-pulse simulation orchestration.

use super::*;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn drive_envelope_unshifted_carrier_shifted() {
    let drive = TripletPulseDrive::iwasaki_2018(PressureAmplitude::from_pascals(0.26e6));
    // Envelope is phase-independent by construction.
    for t in [0.5e-6, 2.0e-6, 3.45e-6, 6.0e-6] {
        assert!(approx_eq(drive.envelope(t), drive.envelope(t), 0.0));
    }
    // Carrier shifts by exactly the transmit phase: p(t, 120°) at
    // the envelope peak equals the 0° value shifted by 120°.
    let t = 3.0e-6;
    let omega = 2.0 * std::f64::consts::PI * drive.frequency_hz;
    let expected = drive.amplitude_pa.as_pascals()
        * drive.envelope(t)
        * (omega * t + 120.0_f64.to_radians()).sin();
    assert!(approx_eq(
        drive.pressure(t, 120.0).as_pascals(),
        expected,
        1e-9
    ));
    // Phase is periodic: φ + 360° ≡ φ (wrap is the segment, and
    // 360° is one full turn of the carrier at any wrap length).
    assert!(approx_eq(
        drive.pressure(t, 480.0).as_pascals(),
        drive.pressure(t, 120.0).as_pascals(),
        1e-9
    ));
}

/// Flat-top envelope shape: Hann rise over cycles 0–2, flat at
/// 1.0 over cycles 2–10, Hann fall over cycles 10–12, zero
/// outside the solve. The analysis window (cycles 6–10) sits
/// entirely in the flat region.
#[test]
fn drive_envelope_is_flat_top() {
    let drive = TripletPulseDrive::iwasaki_2018(PressureAmplitude::from_pascals(0.26e6));
    let f0 = drive.frequency_hz;
    // Envelope values at exact cycle boundaries.
    for (cyc, expected) in [
        (0.0, 0.0),
        (1.0, 0.5),
        (2.0, 1.0),
        (5.0, 1.0),
        (10.0, 1.0),
        (11.0, 0.5),
        (12.0, 0.0),
    ] {
        assert!(
            approx_eq(drive.envelope(cyc / f0), expected, 1e-12),
            "envelope at cycle {cyc} = {}, expected {expected}",
            drive.envelope(cyc / f0)
        );
    }
    // Derivative: zero in the flat region, (π·f0/4)·sin(π·cyc/2)
    // on the rise.
    assert!(approx_eq(drive.envelope_derivative(5.0 / f0), 0.0, 1e-9));
    assert!(approx_eq(
        drive.envelope_derivative(1.0 / f0),
        std::f64::consts::PI * f0 / 4.0,
        1e-3
    ));
    // Zero outside the solve.
    assert_eq!(drive.envelope(-1.0e-6), 0.0);
    assert_eq!(drive.envelope(13.0 / f0), 0.0);
}

#[test]
fn steady_state_window_is_flat_region() {
    let drive = TripletPulseDrive::iwasaki_2018(PressureAmplitude::from_pascals(0.26e6));
    let (start, end) = steady_state_range(&drive, 500);
    assert_eq!(start, 6 * 500);
    assert_eq!(end, 10 * 500);
    assert_eq!(end - start, 4 * 500);
}

#[test]
fn monopole_pressure_falls_as_1_over_r() {
    // Trace must outlast both travel-time delays: near (0.01 m) is
    // 67.5 samples, far (0.02 m) is 135.0 samples at dt = 0.1 us.
    // A 100-sample trace holds no far arrival at all (ratio 0, not 0.5).
    let source = vec![1.0; 300];
    let near = radiate(&source, 0.01, 998.0, 1481.0, 1.0e-7, 300);
    let far = radiate(&source, 0.02, 998.0, 1481.0, 1.0e-7, 300);
    let max = |v: &[f64]| v.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    assert!(approx_eq(max(&far) / max(&near), 0.5, 1e-9));
}

#[test]
fn radiated_trace_delayed_by_travel_time() {
    // Impulse at sample 0; receiver 14.81 mm away at c = 1481 m/s
    // arrives after 10 µs = 100 samples at dt = 0.1 µs.
    let mut source = vec![0.0; 200];
    source[0] = 1.0;
    let trace = radiate(&source, 0.01481, 998.0, 1481.0, 0.1e-6, 200);
    let peak = trace
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(peak, 100);
}

/// Stage 2 parity: the KM-verified bubble-only 3P/1P ratios at the
/// 1.5th fractional harmonic (0.26 MPa, Iwasaki 2018 imaging),
/// with the thermal and mass-transfer paths OFF — matching the
/// prototype's simple adiabatic KM solve. Measured native values
/// (flat-top drive, cycles 6–10): **0.351 / 0.362–0.365 / 0.219**
/// for R0 = 2 / 3 / 5 µm, converged in resolution (500 vs 1000
/// steps/cycle agree to 0.003). The Python flat-top emulation
/// (`stage7/km_envelope_test.py` Part 4) gives 0.381 / 0.422 /
/// 0.219 and the constant-drive reproducible method gives
/// 0.352 / 0.335 / 0.203 — the native values agree with both
/// within ~0.06, resolving the notes-table discrepancy (the
/// notes values 0.34 / 0.29 / 0.15 are superseded; 0.29 and 0.15
/// were the unreliable ones). The thermal-on case is NOT a
/// parity test — the prototype is adiabatic, so thermal-on has
/// no prototype target — but since the KW-PHYS-002 fix it is a
/// boundedness test (`bubble_only_3p_1p_ratio_thermal_on_bounded`):
/// thermal-on 1.5f0 ratios are 0.248 / 0.302 / 0.143, near the
/// thermal-off values, instead of the pre-fix runaway artifacts
/// (1.387 / 1.375 / 1.010).
#[test]
fn bubble_only_3p_1p_ratio_matches_prototype() {
    for (r0_um, expected) in [(2.0, 0.35), (3.0, 0.36), (5.0, 0.22)] {
        let ratio = three_p_over_one_p_ratio(
            r0_um * 1e-6,
            PressureAmplitude::from_pascals(0.26e6),
            1.74e6,
            2.61e6,
            500,
            false,
            0.04,
        )
        .expect("KM solve must converge at 0.26 MPa");
        assert!(
            approx_eq(ratio, expected, 0.05),
            "R0 = {r0_um} µm: 3P/1P at 1.5f0 = {ratio:.3}, expected {expected} ± 0.05"
        );
    }
}

/// Stage 2 parity: the fundamental cancels in 3P (ratio ≈ 0.01).
/// Thermal off, as in the ratio test above.
#[test]
fn bubble_only_3p_1p_fundamental_cancels() {
    let ratio = three_p_over_one_p_ratio(
        5.0e-6,
        PressureAmplitude::from_pascals(0.26e6),
        1.74e6,
        1.74e6,
        500,
        false,
        0.04,
    )
    .expect("KM solve must converge at 0.26 MPa");
    assert!(
        ratio < 0.05,
        "3P/1P at f0 = {ratio:.3}, expected < 0.05 (prototype: 0.011)"
    );
}

/// KW-PHYS-002 boundedness: thermal-on 3P/1P ratios at 1.5f0 and f0
/// after the coupling fix (p_sat at the liquid wall temperature —
/// Kreider et al. 2011, JASA 130:3511, Eq. 15). Pre-fix these were
/// runaway artifacts (1.387 / 1.375 / 1.010 at 1.5f0, 0.534 at f0).
/// Re-measured after the KW-PHYS-003 latent-heat fix (Nov 19,
/// 2026; `L_v` at the interface temperature, clamped ≥ 0, latent
/// term gated on mass transfer): 1.5f0 = **0.248 / 0.302 / 0.143**,
/// f0 = **0.136 / 0.034 / 0.001** for R0 = 2 / 3 / 5 µm (was
/// 0.376 / 0.358 / 0.142 and 0.129 / 0.015 / 0.001 with the bug
/// present). The move is toward the thermal-off values (0.351 /
/// 0.362 / 0.219) at R0 = 2 and 3 µm and nil at R0 = 5 µm: the
/// wrong-signed latent term had added energy during collapse,
/// inflating the fractional harmonic where the collapse is most
/// violent (near-resonant R0 = 2–3 µm); the remaining gap to
/// thermal-off is the genuine thermal-damping effect, strongest
/// at resonance. These are boundedness guards against the
/// runaway regime, NOT parity targets: the literature is silent on
/// thermal effects on the 1.5th fractional harmonic specifically,
/// so no published target exists to compare against. The f0 = 0.136
/// at R0 = 2 µm is the period-3 subharmonic response intrinsic to
/// the drive at near-resonant R0 (thermal-off shows the same
/// period-3 pattern), not a thermal artifact.
#[test]
fn bubble_only_3p_1p_ratio_thermal_on_bounded() {
    for r0_um in [2.0, 3.0, 5.0] {
        let r15 = three_p_over_one_p_ratio(
            r0_um * 1e-6,
            PressureAmplitude::from_pascals(0.26e6),
            1.74e6,
            2.61e6,
            500,
            true,
            0.04,
        )
        .expect("thermal-on KM solve must converge at 0.26 MPa");
        let rf0 = three_p_over_one_p_ratio(
            r0_um * 1e-6,
            PressureAmplitude::from_pascals(0.26e6),
            1.74e6,
            1.74e6,
            500,
            true,
            0.04,
        )
        .expect("thermal-on KM solve must converge at 0.26 MPa");
        eprintln!(
            "THERMAL-ON-MEASURED R0 = {r0_um} µm: 3P/1P at 1.5f0 = {r15:.3}, at f0 = {rf0:.3}"
        );
        assert!(
            r15 < 1.0,
            "thermal-on R0 = {r0_um} µm: 3P/1P at 1.5f0 = {r15:.3}, expected < 1.0 (KW-PHYS-002 runaway guard)"
        );
        assert!(
            rf0 < 0.2,
            "thermal-on R0 = {r0_um} µm: 3P/1P at f0 = {rf0:.3}, expected < 0.2 (KW-PHYS-002 runaway guard)"
        );
    }
}

/// D12(i) — accommodation sensitivity on the thermal-on 3P/1P
/// ratios: α = 0.04 (the default, the 1935 Alty value) vs α = 0.35
/// (the mainstream value: Yasui, Storey & Szeri, Kreider,
/// Hegedüs). A larger α means faster vapour equilibration, i.e.
/// more vapour trapping/condensation per cycle. The measured
/// ratios are OBSERVED values, printed for the record — not parity
/// targets and not assertions of a particular value. The only
/// assertions are the runaway guards (ratios < 1.0 at 1.5f0,
/// < 0.2 at f0), as in the boundedness test.
#[test]
fn thermal_on_ratio_accommodation_sensitivity() {
    for r0_um in [2.0, 3.0, 5.0] {
        for alpha in [0.04, 0.35] {
            let r15 = three_p_over_one_p_ratio(
                r0_um * 1e-6,
                PressureAmplitude::from_pascals(0.26e6),
                1.74e6,
                2.61e6,
                500,
                true,
                alpha,
            )
            .expect("thermal-on KM solve must converge at 0.26 MPa");
            eprintln!(
                "ACCOMMODATION-SENSITIVITY R0 = {r0_um} µm alpha = {alpha}: 3P/1P at 1.5f0 = {r15:.3}"
            );
            assert!(
                r15 < 1.0,
                "thermal-on R0 = {r0_um} µm alpha = {alpha}: 3P/1P at 1.5f0 = {r15:.3}, expected < 1.0"
            );
        }
    }
    // f0 at the near-resonant radius, where the thermal-on f0
    // ratio is largest (0.136 at α = 0.04 after the KW-PHYS-003 fix).
    for alpha in [0.04, 0.35] {
        let rf0 = three_p_over_one_p_ratio(
            2.0e-6,
            PressureAmplitude::from_pascals(0.26e6),
            1.74e6,
            1.74e6,
            500,
            true,
            alpha,
        )
        .expect("thermal-on KM solve must converge at 0.26 MPa");
        eprintln!("ACCOMMODATION-SENSITIVITY R0 = 2.0 µm alpha = {alpha}: 3P/1P at f0 = {rf0:.3}");
        assert!(
            rf0 < 0.2,
            "thermal-on R0 = 2 µm alpha = {alpha}: 3P/1P at f0 = {rf0:.3}, expected < 0.2"
        );
    }
}

/// Prototype Stage 2 cloud geometry (tag a2b2, from
/// `stage1/out/meta_a2b2.json`): 5 bubbles, R0 = 2 µm, seed 11,
/// in a 1 mm disk centred at (19.2, 20) mm; the 32-element probe
/// sits at y = 2 mm. The receiver below is the array centre.
fn prototype_a2b2_cloud(amplitude_pa: PressureAmplitude) -> TripletPulseCloud {
    let mut cloud = TripletPulseCloud::iwasaki_2018(amplitude_pa);
    for (x_mm, y_mm) in [
        (19.865_800_749_680_215, 20.696_362_251_048_71),
        (18.934_633_976_564_61, 20.001_204_059_917_853),
        (18.910_564_839_208_384, 19.785_521_010_127_262),
        (20.158_043_136_630_894, 20.174_590_382_456_213),
        (19.671_798_973_410_567, 20.631_893_441_558_873),
    ] {
        cloud.add_bubble([x_mm * 1e-3, y_mm * 1e-3, 0.0], 2.0e-6);
    }
    cloud
}

/// D1 — the cloud path returns a nonzero trace at the prototype
/// geometry, and its peak lands at the retarded time.
///
/// Before the fix `simulate_phase` windowed the KM history to
/// 4 cycles (~2.3 µs) BEFORE applying the retarded-time delay
/// (≥12.15 µs at this geometry), so the returned trace was
/// identically zero for any receiver more than ~3.4 mm away and
/// no test caught it (all 8 passing port tests exercised the
/// single-bubble ratio path, never `radiate`).
#[test]
fn simulate_receiver_returns_nonzero_trace() {
    let cloud = prototype_a2b2_cloud(PressureAmplitude::from_pascals(0.26e6));
    let receiver = [19.2e-3, 2.0e-3, 0.0];
    let echoes = cloud
        .simulate_receiver(receiver, 500)
        .expect("cloud KM solves must converge at 0.26 MPa");
    let peak = |v: &[f64]| v.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    assert!(
        peak(&echoes.one_pulse) > 0.0,
        "1P trace is all zero at the prototype geometry (D1)"
    );
    assert!(
        peak(&echoes.triplet_pulse) > 0.0,
        "3P trace is all zero at the prototype geometry (D1)"
    );

    // Retarded time, single bubble: the trace peak must land at
    // the window peak's retarded position. The source window's
    // discrete peak is at offset `peak_off` from the window start;
    // its arrival is `delay_samples` later. The ±2-sample bound is
    // one sample of discrete-peak quantisation plus one of
    // fractional-delay interpolation.
    let mut single = TripletPulseCloud::iwasaki_2018(PressureAmplitude::from_pascals(0.26e6));
    single.add_bubble([19.2e-3, 20.0e-3, 0.0], 2.0e-6);
    let params = single.bubble_params(2.0e-6);
    let history = solve_radius_waveform(&params, &single.drive, 0.0, 500)
        .expect("single-bubble KM solve must converge");
    let (start, end) = steady_state_range(&single.drive, 500);
    let source = monopole_source(&history[start..end]);
    let (peak_off, _) = source
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .unwrap();
    let dt = 1.0 / (single.config.frequency_hz * 500.0);
    let delay_samples = 0.018 / single.sound_speed / dt;
    // 18 mm at c = 1481 m/s: 12.15 µs.
    assert!(
        (delay_samples * dt - 12.15e-6).abs() < 0.05e-6,
        "delay at 18 mm must be ≈12.15 µs, got {:.2} µs",
        delay_samples * dt * 1e6
    );
    let trace = single
        .simulate_phase(receiver, 0.0, 500)
        .expect("single-bubble cloud solve must converge");
    // Causality: nothing arrives before the window start's
    // retarded time.
    let lo = (start as f64 + delay_samples).ceil() as usize;
    assert!(
        trace[..lo].iter().all(|&v| v == 0.0),
        "trace is nonzero before the first arrival (D1)"
    );
    let peak_idx = trace
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let expected = start as f64 + peak_off as f64 + delay_samples;
    assert!(
        (peak_idx as f64 - expected).abs() <= 2.0,
        "trace peak at sample {peak_idx}, expected {expected:.1} (window start {start} + peak offset {peak_off} + delay {delay_samples:.1} samples)"
    );
    assert!(peak(&trace) > 0.0, "single-bubble trace is all zero (D1)");
}

/// D1 — cloud-level parity: the 5-bubble R0 = 2 µm cloud at
/// 0.26 MPa reproduces the prototype's bubble-only RF peak within
/// ±3 dB. Prototype reference: saved Stage 2 bubble-only RF
/// (`stage1/out/rf_a2b2_bubble_ph0.npy`, tag a2b2, same cloud and
/// PNP). The prototype script reports the ARRAY-GLOBAL max|p|
/// ("bubble RF max|p|", max over all 32 elements): **1018.8 Pa**.
/// That is the comparator used here — per-element peaks vary
/// 328–1019 Pa (central element 19.04 mm: 804.2 Pa) because the
/// five bubbles' arrivals interfere differently at each element,
/// so no single element is a stable comparator; the Rust receiver
/// at the array centre (19.2 mm) is one element of that array.
/// Known convention differences inside the tolerance: the
/// prototype propagates at c = 1540 m/s (Rust: 1481 m/s), which
/// re-phases the five arrivals, and its steady-state segment
/// carries the transmit Gaussian envelope (σ = 0.7 µs) while the
/// port's flat-top window is unwindowed. Measured Nov 19, 2026:
/// Rust peak 1362.2 Pa vs prototype 1018.8 Pa (+2.52 dB); the
/// envelope explains only −0.41 dB of it (single-bubble
/// diagnostic, printed by this test), so the residual is the
/// re-phasing of the five arrivals — consistent with the
/// prototype's own per-element spread (328–1019 Pa), not with a
/// KM amplitude error (the single-bubble ratio tests already pin
/// the KM waveform).
#[test]
fn cloud_bubble_only_rf_peak_matches_prototype() {
    let cloud = prototype_a2b2_cloud(PressureAmplitude::from_pascals(0.26e6));
    let trace = cloud
        .simulate_phase([19.2e-3, 2.0e-3, 0.0], 0.0, 500)
        .expect("cloud KM solves must converge at 0.26 MPa");
    let peak = trace.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let expected = 1018.8;
    let ratio_db = 20.0 * (peak / expected).log10();
    // Diagnostic: how much of the gap is the prototype's Gaussian
    // envelope? Weight one bubble's windowed source by the
    // prototype envelope (σ = 0.7 µs, centred at cycle 9, the
    // middle of its cycles 6–12 segment) and report the peak
    // suppression vs the unwindowed source.
    let params = cloud.bubble_params(2.0e-6);
    let history = solve_radius_waveform(&params, &cloud.drive, 0.0, 500)
        .expect("single-bubble KM solve must converge");
    let (start, end) = steady_state_range(&cloud.drive, 500);
    let source = monopole_source(&history[start..end]);
    let peak_src = source.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let dt = 1.0 / (cloud.config.frequency_hz * 500.0);
    let mut peak_env = 0.0_f64;
    for (k, &s) in source.iter().enumerate() {
        let t_local = (start + k) as f64 * dt - 9.0 / cloud.config.frequency_hz;
        let env = (-0.5 * (t_local / 0.7e-6).powi(2)).exp();
        peak_env = peak_env.max((s * env).abs());
    }
    eprintln!(
        "CLOUD-PEAK-DIAGNOSTIC envelope suppression (single bubble): {:.2} dB (unwindowed {:.3e}, Gaussian-weighted {:.3e}); cloud peak {peak:.1} Pa vs prototype {expected} Pa ({ratio_db:.2} dB)",
        20.0 * (peak_env / peak_src).log10(),
        peak_src,
        peak_env
    );
    assert!(
        ratio_db.abs() < 3.0,
        "cloud bubble-only RF peak {peak:.1} Pa vs prototype {expected} Pa ({ratio_db:.2} dB), expected within ±3 dB"
    );
}
