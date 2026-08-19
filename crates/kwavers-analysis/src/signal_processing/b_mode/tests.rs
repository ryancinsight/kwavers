//! Value-semantic tests for the B-mode display pipeline.

use super::detection::{envelope, log_compress};
use super::scan_conversion::{CartesianGrid, ScanGeometry, scan_convert};
use super::tgc::TgcConfig;
use aequitas::systems::si::quantities::{Angle, Length};
use aequitas::systems::si::units::{Meter, Radian};
use leto::{Array1, Array2};
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

fn test_geometry() -> ScanGeometry {
    // ±30° sector, 0.5° beams, apex at origin, 0.2 mm range samples.
    ScanGeometry {
        angle_min: Angle::from_unit::<Radian>(-30.0_f64.to_radians()),
        angle_step: Angle::from_unit::<Radian>(0.5_f64.to_radians()),
        radius_offset: Length::from_unit::<Meter>(0.0),
        range_step: Length::from_unit::<Meter>(2e-4),
    }
}

fn test_grid() -> CartesianGrid {
    CartesianGrid {
        width: 200,
        height: 200,
        x_range: (
            Length::from_unit::<Meter>(-0.03),
            Length::from_unit::<Meter>(0.03),
        ),
        z_range: (
            Length::from_unit::<Meter>(0.0),
            Length::from_unit::<Meter>(0.06),
        ),
    }
}

#[test]
fn scan_conversion_places_beam_sample_at_correct_cartesian_pixel() {
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
    let img = scan_convert(beam.view(), test_geometry(), test_grid()).unwrap();
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
    assert!(
        scan_convert(Array2::<f64>::zeros((1, 10)).view(), test_geometry(), test_grid()).is_err()
    );
}

#[test]
fn scan_conversion_rejects_invalid_typed_geometry() {
    let mut geometry = ScanGeometry {
        angle_min: Angle::from_unit::<Radian>(0.0),
        angle_step: Angle::from_unit::<Radian>(0.0),
        radius_offset: Length::from_unit::<Meter>(0.0),
        range_step: Length::from_unit::<Meter>(1e-3),
    };
    let grid = CartesianGrid {
        width: 2,
        height: 2,
        x_range: (
            Length::from_unit::<Meter>(-1.0),
            Length::from_unit::<Meter>(1.0),
        ),
        z_range: (
            Length::from_unit::<Meter>(0.0),
            Length::from_unit::<Meter>(1.0),
        ),
    };
    assert!(scan_convert(Array2::<f64>::zeros((2, 2)).view(), geometry, grid).is_err());

    geometry.angle_step = Angle::from_unit::<Radian>(1.0);
    geometry.radius_offset = Length::from_unit::<Meter>(-1.0);
    assert!(scan_convert(Array2::<f64>::zeros((2, 2)).view(), geometry, grid).is_err());
}

/// Differential oracle for the geometry migration: converting through
/// `ritk_spatial::CurvilinearArray` must reproduce the formulas this module
/// used to implement itself, pixel for pixel.
///
/// The reference below is the pre-migration arithmetic verbatim
/// (`r = hypot(z,x)`, `theta = atan2(x,z)`, `line = (theta - angle_min)/angle_step`,
/// `sample = (r - radius_offset)/range_step`), so the assertion fails if the
/// delegated geometry disagrees anywhere on the raster.
///
/// # Tolerance derivation
///
/// The two paths are not bit-identical, and exact equality would be an
/// analytically wrong assertion. The reference uses `atan2(x, z)` while the
/// geometry uses `atan(x/z)` (ITK's form). For `z > 0` these agree
/// mathematically but not in floating point: the quotient `x/z` is rounded once
/// before `atan` is applied. That rounding is `≤ ε/2` relative, and since
/// `|d atan(u)/du| = 1/(1+u²)`, the angle error is bounded by
/// `(ε/2)·|u|/(1+u²) = (ε/2)·|sin θ cos θ| ≤ ε/4 ≈ 5.6e-17 rad`.
///
/// Dividing by the beam pitch `Δ = 0.5° = 8.727e-3 rad` gives a beam-index error
/// near `1.3e-14`. The bilinear value is Lipschitz in the beam index with
/// constant equal to the largest adjacent-beam difference, `1e3` in this
/// fixture, so the value error is bounded by about `1.3e-11`. The observed
/// worst case is `8e-12`.
///
/// `1e-9` is used: comfortably above that bound, and still ~14 orders below the
/// fixture's own values (up to `1.2e5`). A genuinely mis-indexed pixel differs
/// by `≥ 1` (adjacent sample) or `~1e3` (adjacent beam), so this tolerance
/// cannot mask an indexing defect — which is what the test exists to catch.
#[test]
fn delegated_geometry_matches_the_previous_inline_formulas() {
    let n_lines = 121; // ±30° at 0.5°
    let n_samples = 300;
    // A structured beam field, so a mis-indexed pixel cannot coincidentally match.
    let mut beam = leto::Array2::zeros((n_lines, n_samples));
    for l in 0..n_lines {
        for s in 0..n_samples {
            beam[[l, s]] = (l as f64) * 1.0e3 + (s as f64);
        }
    }
    let got = scan_convert(beam.view(), test_geometry(), test_grid()).expect("conversion");

    // Pre-migration reference, recomputed here independently of the converter.
    let angle_min = -30.0_f64.to_radians();
    let angle_step = 0.5_f64.to_radians();
    let radius_offset = 0.0_f64;
    let range_step = 2e-4_f64;
    let (width, height) = (200_usize, 200_usize);
    let (x_min, z_min) = (-0.03_f64, 0.0_f64);
    let dx = (0.03 - x_min) / (width - 1) as f64;
    let dz = (0.06 - z_min) / (height - 1) as f64;

    let mut checked_interior = 0_usize;
    for row in 0..height {
        let z = z_min + row as f64 * dz;
        for col in 0..width {
            let x = x_min + col as f64 * dx;
            let r = z.hypot(x);
            let theta = x.atan2(z);
            let line = (theta - angle_min) / angle_step;
            let sample = (r - radius_offset) / range_step;

            let mut want = 0.0;
            if line >= 0.0 && sample >= 0.0 {
                let l0 = line.floor() as usize;
                let s0 = sample.floor() as usize;
                if l0 + 1 < n_lines && s0 + 1 < n_samples {
                    let fl = line - l0 as f64;
                    let fs = sample - s0 as f64;
                    want = beam[[l0, s0]] * (1.0 - fl) * (1.0 - fs)
                        + beam[[l0 + 1, s0]] * fl * (1.0 - fs)
                        + beam[[l0, s0 + 1]] * (1.0 - fl) * fs
                        + beam[[l0 + 1, s0 + 1]] * fl * fs;
                    checked_interior += 1;
                }
            }
            let delta = (got[[row, col]] - want).abs();
            assert!(
                delta < 1.0e-9,
                "pixel ({row}, {col}) at (x={x}, z={z}): got {}, want {want}, delta {delta:e}",
                got[[row, col]]
            );
        }
    }
    // Guard against a vacuous pass: the fan must actually cover the raster.
    assert!(
        checked_interior > 5_000,
        "only {checked_interior} pixels landed inside the fan; fixture is not exercising the map"
    );
}
