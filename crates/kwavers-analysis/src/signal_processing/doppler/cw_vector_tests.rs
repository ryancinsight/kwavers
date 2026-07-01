//! Value-semantic tests for continuous-wave Doppler and cross-beam vector flow.

use super::continuous_wave::{ContinuousWaveDoppler, CwDopplerConfig, CwSpectrum};
use super::continuous_wave_vector_flow_fixture;
use super::vector_flow::{VectorFlowEstimator, VectorVelocity};
use std::f64::consts::PI;

// ── Continuous-wave Doppler ───────────────────────────────────────────────────

/// Synthesize a real received tone at `f₀ + f_d` sampled at `fs`.
fn tone(f0: f64, f_d: f64, fs: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (2.0 * PI * (f0 + f_d) * i as f64 / fs).cos())
        .collect()
}

// 2.5 MHz carrier, 20 MHz RF (> 2·f₀), baseband 100 kHz ⇒ decimation 200.
// 409 600 RF samples ⇒ 2048-point baseband for fine velocity resolution.
const CW_FS: f64 = 20e6;
const CW_N: usize = 409_600;

/// Velocity bin width of a spectrum (tolerance unit).
fn dv_of(spec: &CwSpectrum) -> f64 {
    (spec.velocity[1] - spec.velocity[0]).abs()
}

#[test]
fn cw_recovers_velocity_from_doppler_shift() {
    let cfg = CwDopplerConfig::new(2.5e6, CW_FS, 0.0);
    let cw = ContinuousWaveDoppler::new(cfg);
    let v_true = 1.2;
    let f_d = 2.0 * v_true * cfg.center_frequency / cfg.sound_speed;
    let rf = tone(cfg.center_frequency, f_d, cfg.sampling_rate, CW_N);

    let spec = cw.spectrum(&rf).unwrap();
    let v_peak = spec.peak_velocity();
    let dv = dv_of(&spec);
    assert!(
        (v_peak - v_true).abs() < 3.0 * dv,
        "peak velocity {v_peak} vs truth {v_true} (dv={dv})"
    );
}

#[test]
fn cw_resolves_velocity_beyond_pulsed_wave_nyquist() {
    // PW at PRF 5 kHz, 2.5 MHz has Nyquist velocity ≈ 0.77 m/s; CW resolves
    // 2.0 m/s without aliasing (baseband Nyquist ≈ 15 m/s here).
    let cfg = CwDopplerConfig::new(2.5e6, CW_FS, 0.0);
    let cw = ContinuousWaveDoppler::new(cfg);
    let pw_prf = 5e3;
    let v_nyquist_pw = pw_prf * cfg.sound_speed / (4.0 * cfg.center_frequency);
    let v_true = 2.0;
    assert!(v_true > v_nyquist_pw, "test premise: above PW Nyquist");

    let f_d = 2.0 * v_true * cfg.center_frequency / cfg.sound_speed;
    let rf = tone(cfg.center_frequency, f_d, cfg.sampling_rate, CW_N);
    let spec = cw.spectrum(&rf).unwrap();
    let v_peak = spec.peak_velocity();
    assert!(
        (v_peak - v_true).abs() < 3.0 * dv_of(&spec),
        "v_peak {v_peak} vs {v_true}"
    );
}

#[test]
fn cw_sign_encodes_flow_direction() {
    let cfg = CwDopplerConfig::new(2.5e6, CW_FS, 0.0);
    let cw = ContinuousWaveDoppler::new(cfg);
    let v_true = -0.8; // away from transducer ⇒ negative f_d
    let f_d = 2.0 * v_true * cfg.center_frequency / cfg.sound_speed;
    let rf = tone(cfg.center_frequency, f_d, cfg.sampling_rate, CW_N);
    let spec = cw.spectrum(&rf).unwrap();
    let v_peak = spec.peak_velocity();
    assert!(
        v_peak < 0.0,
        "expected negative (away) velocity, got {v_peak}"
    );
    assert!(
        (v_peak - v_true).abs() < 3.0 * dv_of(&spec),
        "v_peak {v_peak} vs {v_true}"
    );
}

#[test]
fn cw_empty_signal_is_rejected() {
    let cw = ContinuousWaveDoppler::new(CwDopplerConfig::new(2.5e6, CW_FS, 0.0));
    assert!(cw.spectrum(&[]).is_err());
}

// ── Cross-beam vector flow ────────────────────────────────────────────────────

/// Unit beam direction tilted `theta` from the axial (z) axis: (sinθ, cosθ).
fn beam(theta: f64) -> [f64; 2] {
    [theta.sin(), theta.cos()]
}

#[test]
fn two_beams_recover_vector_exactly() {
    let v = VectorVelocity { vx: 0.3, vz: -0.5 };
    let dirs = [beam(0.35), beam(-0.35)];
    // Forward project: v_i = v · d̂_i.
    let projected: Vec<f64> = dirs.iter().map(|d| v.vx * d[0] + v.vz * d[1]).collect();

    let est = VectorFlowEstimator::new(&dirs).unwrap();
    let recovered = est.estimate(&projected).unwrap();
    assert!((recovered.vx - v.vx).abs() < 1e-12, "vx {}", recovered.vx);
    assert!((recovered.vz - v.vz).abs() < 1e-12, "vz {}", recovered.vz);
}

#[test]
fn overdetermined_three_beams_recover_vector() {
    let v = VectorVelocity { vx: -0.4, vz: 0.9 };
    let dirs = [beam(0.4), beam(0.0), beam(-0.4)];
    let projected: Vec<f64> = dirs.iter().map(|d| v.vx * d[0] + v.vz * d[1]).collect();
    let est = VectorFlowEstimator::new(&dirs).unwrap();
    let r = est.estimate(&projected).unwrap();
    assert!((r.vx - v.vx).abs() < 1e-12 && (r.vz - v.vz).abs() < 1e-12);
}

#[test]
fn magnitude_and_angle_are_consistent() {
    let v = VectorVelocity { vx: 0.3, vz: 0.4 };
    assert!((v.magnitude() - 0.5).abs() < 1e-12);
    // atan2(0.3, 0.4) from the axial axis.
    assert!((v.angle() - 0.3_f64.atan2(0.4)).abs() < 1e-12);
}

#[test]
fn collinear_beams_are_rejected() {
    // Two beams along the same axis cannot resolve the lateral component.
    let dirs = [beam(0.2), beam(0.2)];
    assert!(VectorFlowEstimator::new(&dirs).is_err());
}

#[test]
fn single_beam_is_rejected() {
    assert!(VectorFlowEstimator::new(&[beam(0.3)]).is_err());
}

#[test]
fn projected_length_mismatch_is_rejected() {
    let est = VectorFlowEstimator::new(&[beam(0.3), beam(-0.3)]).unwrap();
    assert!(est.estimate(&[0.1]).is_err());
}

#[test]
fn continuous_wave_vector_flow_fixture_uses_rust_doppler_kernels() {
    let fixture = continuous_wave_vector_flow_fixture(
        2.5e6,
        CW_FS,
        100e3,
        2.0,
        5e3,
        1540.0,
        2048,
        VectorVelocity { vx: 0.35, vz: 0.55 },
        &[25.0_f64.to_radians(), (-25.0_f64).to_radians()],
    )
    .unwrap();

    let peak_index = fixture
        .cw_power
        .iter()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
        .map(|(index, _)| index)
        .unwrap();
    let v_peak = fixture.cw_velocity_m_s[peak_index];
    let dv = (fixture.cw_velocity_m_s[1] - fixture.cw_velocity_m_s[0]).abs();
    assert!(
        (v_peak - 2.0).abs() < 3.0 * dv,
        "CW peak {v_peak} vs 2.0 m/s with dv {dv}"
    );
    assert!(
        2.0 > fixture.pulsed_wave_nyquist_velocity_m_s,
        "test premise: CW jet exceeds PW Nyquist"
    );
    assert!(fixture.vector_error_m_s < 1e-12);
    assert_eq!(fixture.beam_directions.len(), 2);
}

#[test]
fn continuous_wave_vector_flow_fixture_rejects_invalid_domains() {
    assert!(continuous_wave_vector_flow_fixture(
        2.5e6,
        CW_FS,
        100e3,
        2.0,
        5e3,
        1540.0,
        0,
        VectorVelocity { vx: 0.35, vz: 0.55 },
        &[25.0_f64.to_radians(), (-25.0_f64).to_radians()],
    )
    .is_err());
    assert!(continuous_wave_vector_flow_fixture(
        2.5e6,
        CW_FS,
        100e3,
        2.0,
        5e3,
        1540.0,
        2048,
        VectorVelocity { vx: 0.35, vz: 0.55 },
        &[0.25, 0.25],
    )
    .is_err());
}
