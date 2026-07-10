//! Value-semantic tests for the B-mode display pipeline.

use super::detection::{envelope, log_compress};
use super::scan_conversion::{CartesianGrid, ScanConverter, ScanGeometry};
use super::tgc::TgcConfig;
use leto::{
    Array1,
    Array2,
};
use std::f64::consts::PI;

// ── Time-gain compensation ────────────────────────────────────────────────────

fn tgc_config() -> TgcConfig {
    TgcConfig {
        attenuation_db_cm_mhz: 0.5,
        frequency_mhz: 5.0,
        sound_speed: 1540.0,
        sampling_rate: 40e6,
    }
}

#[test]
fn tgc_gain_matches_attenuation_law() {
    let cfg = tgc_config();
    // At sample i, gain_dB = 2·a₀·f·z_cm.
    let i = 5000;
    let z_cm = cfg.depth_m(i) * 100.0;
    let expected_db = 2.0 * 0.5 * 5.0 * z_cm;
    let expected_gain = 10.0_f64.powf(expected_db / 20.0);
    assert!((cfg.gain(i) - expected_gain).abs() < 1e-9 * expected_gain);
    assert!((cfg.gain(0) - 1.0).abs() < 1e-12, "depth 0 ⇒ unit gain");
}

#[test]
fn tgc_flattens_attenuated_echoes() {
    let cfg = tgc_config();
    let n = 8000;
    // Equal reflectors attenuated by the round-trip law: a(i) = 10^(−A(z)/20).
    let attenuated = Array1::from_shape_fn(n, |[i]| {
        let z_cm = cfg.depth_m(i) * 100.0;
        let a_db = 2.0 * cfg.attenuation_db_cm_mhz * cfg.frequency_mhz * z_cm;
        10.0_f64.powf(-a_db / 20.0)
    });
    let corrected = cfg.apply(&attenuated).unwrap();
    // After TGC every reflector reads ≈ 1.0 regardless of depth.
    for i in (0..n).step_by(500) {
        assert!(
            (corrected[i] - 1.0).abs() < 1e-9,
            "corrected[{i}] = {}",
            corrected[i]
        );
    }
}

#[test]
fn tgc_rejects_invalid_config() {
    let bad = TgcConfig {
        sound_speed: 0.0,
        ..tgc_config()
    };
    assert!(bad.apply(&Array1::zeros(10)).is_err());
}

// ── Envelope + log compression ────────────────────────────────────────────────

#[test]
fn envelope_of_tone_equals_amplitude() {
    let n = 1024;
    let amp = 2.5;
    let f = 0.1; // cycles/sample
    let rf = Array1::from_shape_fn(n, |[i]| amp * (2.0 * PI * f * i as f64).cos());
    let env = envelope(&rf);
    // Interior envelope of a pure tone is the amplitude (edges ring from FFT).
    for i in 100..n - 100 {
        assert!((env[i] - amp).abs() < 0.05 * amp, "env[{i}] = {}", env[i]);
    }
}

#[test]
fn log_compression_maps_dynamic_range() {
    let dr = 40.0;
    // Peak 1.0, a −40 dB point (0.01), and a sub-floor point (1e-4).
    let env = Array1::from(vec![1.0, 0.01, 1e-4]);
    let out = log_compress(&env, dr).unwrap();
    assert!((out[0] - 1.0).abs() < 1e-12, "peak → 1");
    assert!(out[1].abs() < 1e-9, "−40 dB → 0 (floor)");
    assert!(out[2].abs() < 1e-12, "below floor → clamped 0");
    // A −20 dB point (0.1) maps to the middle of the range.
    let mid = log_compress(&Array1::from(vec![1.0, 0.1]), dr).unwrap();
    assert!((mid[1] - 0.5).abs() < 1e-9, "−20 dB → 0.5");
}

#[test]
fn log_compress_rejects_bad_range() {
    assert!(log_compress(&Array1::from(vec![1.0]), 0.0).is_err());
}

// ── Scan conversion ───────────────────────────────────────────────────────────

fn converter() -> ScanConverter {
    // ±30° sector, 0.5° beams, apex at origin, 0.2 mm range samples.
    let geometry = ScanGeometry {
        angle_min: -30.0_f64.to_radians(),
        angle_step: 0.5_f64.to_radians(),
        radius_offset: 0.0,
        range_step: 2e-4,
    };
    let grid = CartesianGrid {
        width: 200,
        height: 200,
        x_range: (-0.03, 0.03),
        z_range: (0.0, 0.06),
    };
    ScanConverter::new(geometry, grid).unwrap()
}

#[test]
fn scan_conversion_places_beam_sample_at_correct_cartesian_pixel() {
    let sc = converter();
    let n_lines = 121; // -30..30 step 0.5
    let n_samples = 300;
    let mut beam = Array2::zeros((n_lines, n_samples));
    // Bright patch on the center beam (θ = 0) at a known range.
    let line = 60; // (0 − (−30))/0.5 = 60 ⇒ θ = 0
    let sample = 200; // r = 200·0.2 mm = 0.04 m straight down
    for dl in 0..2 {
        for ds in 0..2 {
            beam[[line + dl, sample + ds]] = 1.0;
        }
    }
    let img = sc.convert(beam.view()).unwrap();
    // Expected Cartesian location: x = 0, z = 0.04 m.
    let dz = 0.06_f64 / 199.0;
    let row = (0.04_f64 / dz).round() as usize;
    let col = 100; // x = 0 is the middle column
    assert!(
        img[[row, col]] > 0.5,
        "expected bright at center beam, got {}",
        img[[row, col]]
    );
    // A pixel well outside the sector (top corner, |θ| ≫ 30°) is background.
    assert!(img[[1, 0]].abs() < 1e-12, "outside-sector pixel must be 0");
}

#[test]
fn scan_conversion_rejects_degenerate_beam_grid() {
    let sc = converter();
    assert!(sc.convert(Array2::<f64>::zeros((1, 10)).view()).is_err());
}
