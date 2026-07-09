//! Value-semantic tests for thermal strain imaging.
//!
//! Coverage: thermoacoustic coefficient sign/magnitude, strain↔temperature
//! inversion round-trip, NCC sub-sample displacement recovery on synthetic RF,
//! exact least-squares strain on a displacement ramp, an end-to-end
//! uniformly-heated-block reconstruction against a known ΔT, and the negative /
//! boundary paths (singular coefficient, even window, dimension mismatch).

use super::config::ThermalStrainConfig;
use super::strain::least_squares_strain;
use super::tracking::{track_line_samples, TrackingParams};
use super::{ThermalStrainImager, ThermalStrainResult};
use leto::{
    Array1,
    Array3,
    ArrayView1,
};

// ── Deterministic synthetic RF generation (no RNG; reproducible) ──────────────

/// Small linear-congruential generator for reproducible pseudo-random speckle.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    // Map high 32 bits to (-1, 1).
    ((*state >> 32) as f64 / u32::MAX as f64) * 2.0 - 1.0
}

/// Band-limited speckle RF line: pseudo-random reflectivity modulated by a
/// carrier, with two-sample correlation so the NCC peak is well-defined.
fn synthetic_rf(nz: usize, seed: u64, samples_per_cycle: f64) -> Array1<f64> {
    let mut state = seed;
    let raw: Vec<f64> = (0..nz).map(|_| lcg(&mut state)).collect();
    let mut rf = Array1::zeros(nz);
    for k in 0..nz {
        // Two-tap moving average → finite correlation length.
        let smooth = if k == 0 {
            raw[0]
        } else {
            0.5 * (raw[k] + raw[k - 1])
        };
        let phase = 2.0 * std::f64::consts::PI * k as f64 / samples_per_cycle;
        rf[k] = smooth * phase.cos();
    }
    rf
}

/// Linear interpolation of `line` at a (possibly fractional) index.
fn interp(line: ArrayView1<f64>, idx: f64) -> f64 {
    let n = line.len();
    if idx <= 0.0 {
        return line[0];
    }
    if idx >= (n - 1) as f64 {
        return line[n - 1];
    }
    let i0 = idx.floor() as usize;
    let frac = idx - i0 as f64;
    line[i0] * (1.0 - frac) + line[i0 + 1] * frac
}

// ── Configuration / coefficient algebra ───────────────────────────────────────

#[test]
fn soft_tissue_coefficient_is_negative_and_physical() {
    let cfg = ThermalStrainConfig::soft_tissue();
    // β_c = (1/c₀)(dc/dT) = 2.0 / 1540 ≈ 1.299e-3 /°C
    let beta = cfg.sound_speed_coefficient();
    assert!((beta - 2.0 / 1540.0).abs() < 1e-9, "β_c = {beta}");
    // k_T = α_th − β_c = 3.0e-4 − 1.299e-3 < 0 for water-based tissue.
    let k_t = cfg.combined_coefficient();
    assert!(
        k_t < 0.0,
        "water-based tissue must have negative k_T, got {k_t}"
    );
    assert!((k_t - (3.0e-4 - 2.0 / 1540.0)).abs() < 1e-12);
}

#[test]
fn lipid_tissue_coefficient_flips_sign() {
    // Fat: dc/dT ≈ −5 m/s/°C (Bamber & Hill 1979) → k_T > 0.
    let cfg = ThermalStrainConfig {
        dc_dt: -5.0,
        ..ThermalStrainConfig::soft_tissue()
    };
    assert!(cfg.combined_coefficient() > 0.0);
}

#[test]
fn temperature_from_strain_round_trip() {
    let cfg = ThermalStrainConfig::soft_tissue();
    let delta_t = 7.5_f64;
    let strain = cfg.combined_coefficient() * delta_t;
    let recovered = cfg.temperature_from_strain(strain);
    assert!((recovered - delta_t).abs() < 1e-9, "recovered {recovered}");
}

#[test]
fn validate_rejects_even_window_and_singular_coefficient() {
    let even = ThermalStrainConfig {
        strain_window: 10,
        ..ThermalStrainConfig::soft_tissue()
    };
    assert!(even.validate().is_err());

    // α_th = β_c ⇒ k_T = 0 ⇒ singular inversion.
    let singular = ThermalStrainConfig {
        thermal_expansion: 2.0 / 1540.0,
        ..ThermalStrainConfig::soft_tissue()
    };
    assert!(singular.validate().is_err());
}

// ── NCC displacement tracking ─────────────────────────────────────────────────

#[test]
fn ncc_recovers_known_subsample_shift() {
    let nz = 256;
    let reference = synthetic_rf(nz, 0x1234_5678, 5.0);
    let true_shift = 1.6_f64; // samples; post[k] = ref[k − shift]
    let tracked: Array1<f64> =
        Array1::from_shape_fn([nz], |k| interp(reference.view(), k as f64 - true_shift));

    let params = TrackingParams {
        window_half: 10,
        max_lag: 5,
    };
    let disp = track_line_samples(reference.view(), tracked.view(), params);

    // Average over the well-conditioned interior.
    let guard = params.window_half + params.max_lag;
    let interior: Vec<f64> = disp[guard..nz - guard].to_vec();
    let mean = interior.iter().sum::<f64>() / interior.len() as f64;
    assert!(
        (mean - true_shift).abs() < 0.15,
        "mean recovered shift {mean} vs truth {true_shift}"
    );
}

// ── Least-squares strain ──────────────────────────────────────────────────────

#[test]
fn least_squares_strain_recovers_exact_slope() {
    // u(z) = s·z with s = 0.01, dz = 1e-4 m ⇒ strain = 0.01 everywhere.
    let (nx, ny, nz) = (1, 1, 64);
    let dz = 1e-4;
    let slope = 0.01;
    let disp = Array3::from_shape_fn([nx, ny, nz], |(_, _, k)| slope * (k as f64 * dz));
    let strain = least_squares_strain(&disp, dz, 7);
    // Interior points must recover the slope to machine precision.
    for k in 5..nz - 5 {
        assert!(
            (strain[[0, 0, k]] - slope).abs() < 1e-10,
            "strain[{k}] = {}",
            strain[[0, 0, k]]
        );
    }
}

// ── End-to-end uniformly heated block ─────────────────────────────────────────

#[test]
fn reconstructs_uniform_temperature_change() {
    let cfg = ThermalStrainConfig::soft_tissue();
    let sampling_rate = 40e6; // 40 MHz RF
    let params = TrackingParams {
        window_half: 12,
        max_lag: 6,
    };
    let imager = ThermalStrainImager::new(cfg, params, sampling_rate).unwrap();

    let (nx, ny, nz) = (3, 3, 384);
    let delta_t_true = 6.0_f64; // °C, uniform
    let k_t = cfg.combined_coefficient();

    // Build pre/post volumes: post[k] = ref[k − d(k)], d(k) = u(z)/dz = k_T·ΔT·k.
    let mut reference = Array3::zeros([nx, ny, nz]);
    let mut tracked = Array3::zeros([nx, ny, nz]);
    for i in 0..nx {
        for j in 0..ny {
            let seed = 0xABCD_0000u64 ^ ((i as u64) << 16) ^ (j as u64);
            let line = synthetic_rf(nz, seed, 5.0);
            for k in 0..nz {
                reference[[i, j, k]] = line[k];
                let d = k_t * delta_t_true * k as f64; // sample shift (negative)
                tracked[[i, j, k]] = interp(line.view(), k as f64 - d);
            }
        }
    }

    let ThermalStrainResult {
        temperature_change,
        strain,
        ..
    } = imager
        .reconstruct_temperature(&reference, &tracked)
        .unwrap();

    // Average ΔT and strain over a central interior block (avoid axial/lateral
    // edges). Pixel-wise NCC jitter cancels in the mean; the bias is what the
    // estimator is responsible for.
    let guard = params.window_half + params.max_lag + cfg.strain_window;
    let mut sum_dt = 0.0;
    let mut sum_strain = 0.0;
    let mut count = 0usize;
    for i in 0..nx {
        for j in 0..ny {
            for k in guard..nz - guard {
                sum_dt += temperature_change[[i, j, k]];
                sum_strain += strain[[i, j, k]];
                count += 1;
            }
        }
    }
    let mean_dt = sum_dt / count as f64;
    let mean_strain = sum_strain / count as f64;
    assert!(
        (mean_dt - delta_t_true).abs() < 1.0,
        "mean recovered ΔT {mean_dt} °C vs truth {delta_t_true} °C"
    );
    // Mean thermal strain sign must match the (negative) thermoacoustic
    // coefficient for water-based tissue, with magnitude near k_T·ΔT.
    let strain_truth = k_t * delta_t_true;
    assert!(
        mean_strain < 0.0,
        "expected negative mean strain, got {mean_strain}"
    );
    assert!(
        (mean_strain - strain_truth).abs() < 0.25 * strain_truth.abs(),
        "mean strain {mean_strain} vs truth {strain_truth}"
    );
}

// ── Negative / boundary paths ─────────────────────────────────────────────────

#[test]
fn dimension_mismatch_is_rejected() {
    let imager = ThermalStrainImager::new(
        ThermalStrainConfig::soft_tissue(),
        TrackingParams::default(),
        40e6,
    )
    .unwrap();
    let a = Array3::zeros([2, 2, 32]);
    let b = Array3::zeros([2, 2, 16]);
    assert!(imager.reconstruct_temperature(&a, &b).is_err());
}

#[test]
fn no_heating_gives_zero_displacement_and_temperature() {
    let cfg = ThermalStrainConfig::soft_tissue();
    let sampling_rate = 40e6;
    let params = TrackingParams {
        window_half: 12,
        max_lag: 6,
    };
    let imager = ThermalStrainImager::new(cfg, params, sampling_rate).unwrap();
    let dz = cfg.axial_sample_spacing(sampling_rate);

    // Identical broadband speckle volumes ⇒ no heating. The unique correlation
    // maximum is at lag 0, so the integer displacement is exactly zero
    // everywhere; only bounded sub-sample residual remains.
    let (nx, ny, nz) = (2, 2, 256);
    let mut rf = Array3::zeros([nx, ny, nz]);
    for i in 0..nx {
        for j in 0..ny {
            let line = synthetic_rf(nz, 0x55AA_0000u64 ^ ((i as u64) << 8) ^ j as u64, 5.0);
            for k in 0..nz {
                rf[[i, j, k]] = line[k];
            }
        }
    }
    let result = imager.reconstruct_temperature(&rf, &rf).unwrap();

    // Integer shift is zero everywhere: |u|/dz < 0.5 sample.
    let max_samples = result
        .displacement
        .iter()
        .fold(0.0_f64, |m, &v| m.max((v / dz).abs()));
    assert!(
        max_samples < 0.5,
        "max |displacement| = {max_samples} samples"
    );

    // No systematic temperature bias.
    let mean_dt = result.temperature_change.sum() / result.temperature_change.len() as f64;
    assert!(mean_dt.abs() < 0.5, "mean ΔT bias = {mean_dt} °C");
}

#[test]
fn uniform_displacement_yields_exactly_zero_strain() {
    // The strain→temperature stage in isolation: a spatially constant (DC)
    // apparent displacement has zero gradient, hence exactly zero thermal strain
    // and zero ΔT, independent of tracking.
    use super::strain::least_squares_strain;
    let disp = Array3::from_elem((2, 2, 64), 3.7e-6_f64);
    let strain = least_squares_strain(&disp, 1e-5, 9);
    let max_abs = strain.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    assert!(max_abs < 1e-15, "max |strain| = {max_abs}");
}
