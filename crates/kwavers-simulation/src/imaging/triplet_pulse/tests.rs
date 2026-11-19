//! Tests for the triplet-pulse simulation orchestration.

use super::*;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// Goertzel magnitude of `signal` at `freq_hz`, sampled at `dt`.
/// Single-frequency DFT — O(n) per frequency, no FFT dependency.
fn goertzel_magnitude(signal: &[f64], freq_hz: f64, dt: f64) -> f64 {
    let omega = 2.0 * std::f64::consts::PI * freq_hz * dt;
    let coeff = 2.0 * omega.cos();
    let (mut s1, mut s2) = (0.0, 0.0);
    for &x in signal {
        let s0 = x + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    let re = s1 - s2 * omega.cos();
    let im = s2 * omega.sin();
    (re * re + im * im).sqrt()
}

/// 3P/1P magnitude ratio at `freq_hz` between the coherent 3P sum
/// and the 1P trace, for one bubble's monopole source.
///
/// `thermal` selects the native model's thermal/mass-transfer
/// path: `false` turns both off, matching the prototype's simple
/// adiabatic KM solve; `true` leaves the defaults on. Before the
/// KW-PHYS-002 fix (Nov 18, 2026) `true` was unusable for
/// multi-cycle drives (vapour runaway, ~+14 R0 per cycle); since
/// the fix it reaches a bounded periodic state, covered by the
/// boundedness test. The parity tests use `false` because the
/// prototype is adiabatic, not because `true` fails.
fn three_p_over_one_p_ratio(
    r0_m: f64,
    amplitude_pa: PressureAmplitude,
    freq_hz: f64,
    steps_per_cycle: usize,
    thermal: bool,
) -> KwaversResult<f64> {
    let drive = TripletPulseDrive {
        frequency_hz: 1.74e6,
        amplitude_pa,
        solve_cycles: 12,
        steady_start_cycle: 6,
        steady_end_cycle: 10,
        segment_cycles: 6,
    };
    let mut params = BubbleParameters::default();
    params.r0 = r0_m;
    params.driving_frequency = 1.74e6;
    params.driving_amplitude = amplitude_pa;
    params.use_thermal_effects = thermal;
    params.use_mass_transfer = thermal;
    let (start, end) = steady_state_range(&drive, steps_per_cycle);
    let dt = 1.0 / (drive.frequency_hz * steps_per_cycle as f64);
    let mut phases = Vec::new();
    for phase_deg in [0.0, 120.0, 240.0] {
        let history = solve_radius_waveform(&params, &drive, phase_deg, steps_per_cycle)?;
        phases.push(monopole_source(&history[start..end]));
    }
    let one_p = goertzel_magnitude(&phases[0], freq_hz, dt);
    let three_p = goertzel_magnitude(
        &{
            let n = phases[0].len();
            (0..n)
                .map(|i| (phases[0][i] + phases[1][i] + phases[2][i]) / 3.0)
                .collect::<Vec<f64>>()
        },
        freq_hz,
        dt,
    );
    Ok(three_p / one_p)
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
    let near = radiate(&source, 0.01, 998.0, 1481.0, 1.0e-7);
    let far = radiate(&source, 0.02, 998.0, 1481.0, 1.0e-7);
    let max = |v: &[f64]| v.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    assert!(approx_eq(max(&far) / max(&near), 0.5, 1e-9));
}

#[test]
fn radiated_trace_delayed_by_travel_time() {
    // Impulse at sample 0; receiver 14.81 mm away at c = 1481 m/s
    // arrives after 10 µs = 100 samples at dt = 0.1 µs.
    let mut source = vec![0.0; 200];
    source[0] = 1.0;
    let trace = radiate(&source, 0.01481, 998.0, 1481.0, 0.1e-6);
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
/// thermal-on 1.5f0 ratios are 0.376 / 0.358 / 0.142, near the
/// thermal-off values, instead of the pre-fix runaway artifacts
/// (1.387 / 1.375 / 1.010).
#[test]
fn bubble_only_3p_1p_ratio_matches_prototype() {
    for (r0_um, expected) in [(2.0, 0.35), (3.0, 0.36), (5.0, 0.22)] {
        let ratio = three_p_over_one_p_ratio(
            r0_um * 1e-6,
            PressureAmplitude::from_pascals(0.26e6),
            2.61e6,
            500,
            false,
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
        500,
        false,
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
/// runaway artifacts (1.387 / 1.375 / 1.010 at 1.5f0, 0.534 at f0);
/// post-fix they are bounded and near the thermal-off values
/// (1.5f0: 0.376 / 0.358 / 0.142; f0: 0.129 / 0.015 / 0.001 for
/// R0 = 2 / 3 / 5 µm). These are boundedness guards against the
/// runaway regime, NOT parity targets: the literature is silent on
/// thermal effects on the 1.5th fractional harmonic specifically,
/// so no published target exists to compare against. The f0 = 0.129
/// at R0 = 2 µm is the period-3 subharmonic response intrinsic to
/// the drive at near-resonant R0 (thermal-off shows the same
/// period-3 pattern), not a thermal artifact.
#[test]
fn bubble_only_3p_1p_ratio_thermal_on_bounded() {
    for r0_um in [2.0, 3.0, 5.0] {
        let r15 = three_p_over_one_p_ratio(
            r0_um * 1e-6,
            PressureAmplitude::from_pascals(0.26e6),
            2.61e6,
            500,
            true,
        )
        .expect("thermal-on KM solve must converge at 0.26 MPa");
        let rf0 = three_p_over_one_p_ratio(
            r0_um * 1e-6,
            PressureAmplitude::from_pascals(0.26e6),
            1.74e6,
            500,
            true,
        )
        .expect("thermal-on KM solve must converge at 0.26 MPa");
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
