//! Differential test: PWLS (inverse-noise-variance) data weighting — the
//! low-dose-CT lesson — makes the FWI reconstruction robust to a subset of
//! high-noise "bad" channels, where the default unweighted L2 misfit is corrupted.
//!
//! Setup: the same single-shot anomaly-recovery problem as `lbfgs.rs` but with a
//! wide transmission aperture, and the "observed" data is contaminated by
//! **heteroscedastic** noise — a minority of receivers (every 4th) are 10× noisier
//! (faulty/poorly-coupled or skull-shadowed channels) than the rest. Both
//! inversions run on the **same** noisy data; only
//! the data weighting differs. PWLS estimates each trace's noise variance from a
//! quiet pre-first-arrival window and down-weights the bad channels (Sauer &
//! Bouman 1993; Thibault et al. 2007), so its reconstruction is closer to the
//! truth at the anomaly than the equally-weighted L2 reconstruction.

use super::super::{FwiGeometry, FwiProcessor};
use crate::inverse::reconstruction::seismic::DataWeighting;
use crate::inverse::seismic::parameters::{FwiParameters, RegularizationParameters};
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_grid::Grid;
use kwavers_source::{GridSource, SourceMode};
use leto::{
    Array2,
    Array3,
};

/// Single-shot problem with a long quiet pre-arrival window (source at `ix=1`,
/// receivers at `ix=6`) so the per-trace noise variance is estimable.
fn build_problem() -> (Grid, FwiGeometry, FwiParameters, Array3<f64>, Array3<f64>) {
    let (nx, ny, nz) = (8usize, 8, 8);
    let dx = 1e-3;
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).expect("grid");
    let dims = (nx, ny, nz);
    let c0 = SOUND_SPEED_WATER_SIM;
    let initial = Array3::from_elem(dims, c0);
    let mut truth = initial.clone();
    for ([ix, iy, iz], value) in truth.indexed_iter_mut().expect("invariant: owned array yields indexed iterator") {
        let r2 = (ix as f64 - 3.5).powi(2) + (iy as f64 - 3.5).powi(2) + (iz as f64 - 3.5).powi(2);
        *value += 60.0 * (-r2 / 3.0).exp();
    }
    // Wide transmission aperture (full ix=6 face) so down-weighting a *minority*
    // of bad channels still leaves redundant coverage — the regime in which the
    // CT/MBIR statistical-weighting lesson applies.
    let mut sensor_mask = Array3::from_elem(dims, false);
    for iy in 1..7 {
        for iz in 1..7 {
            sensor_mask[[6, iy, iz]] = true;
        }
    }
    let nt = 96usize;
    let dt = 1e-7;
    let mut p_mask = Array3::from_elem(dims, 0.0_f64);
    p_mask[[1, 4, 4]] = 1.0;
    let mut p_signal = Array2::zeros((1, nt));
    for t in 0..24 {
        let phase = (t as f64) * 0.35;
        p_signal[[0, t]] = (-phase * phase / 9.0).exp() * (2.0 * phase).sin();
    }
    let source = GridSource {
        p0: None,
        u0: None,
        p_mask: Some(p_mask),
        p_signal: Some(p_signal),
        p_mode: SourceMode::Additive,
        u_mask: None,
        u_signal: None,
        u_mode: SourceMode::default(),
    };
    let geometry = FwiGeometry::new(source, sensor_mask);
    let parameters = FwiParameters {
        nt,
        dt,
        frequency: 5e5,
        max_iterations: 8,
        step_size: 50.0,
        tolerance: 1e-6,
        source_mute_radius: 3,
        regularization: RegularizationParameters {
            tikhonov_weight: 0.0,
            tv_weight: 0.0,
            directional_tv_weight: 0.0,
            directional_tv_adaptive: false,
            smoothness_weight: 0.0,
        },
        ..FwiParameters::default()
    };
    (grid, geometry, parameters, truth, initial)
}

/// PWLS recovers the anomaly more accurately than unweighted L2 when a subset of
/// channels is much noisier — the transcranial analogue of CT's low-photon-ray
/// down-weighting.
/// # Panics
/// - Panics on any assertion failure or solver error.
#[test]
fn pwls_is_robust_to_bad_channels_vs_unweighted_l2() {
    let (grid, geometry, parameters, truth, initial) = build_problem();

    // Clean observed data, then contaminate with heteroscedastic noise.
    let base = FwiProcessor::new(parameters.clone());
    let (clean, _hist) = base
        .forward_model(&truth, &geometry, &grid)
        .expect("observed forward");
    let [n_rx, nt] = clean.shape();
    let rms = (clean.iter().map(|v| v * v).sum::<f64>() / (clean.len()) as f64).sqrt();

    // A minority (every 4th receiver) is 10× noisier (variance 100×). Same
    // realisation feeds both inversions.
    let mut lcg: u64 = 0x2545_F491_4F6C_DD1D;
    let mut gauss = || {
        // Box–Muller from a deterministic LCG.
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u1 = (((lcg >> 11) as f64) + 0.5) / ((1u64 << 53) as f64);
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u2 = (((lcg >> 11) as f64) + 0.5) / ((1u64 << 53) as f64);
        (-2.0 * u1.max(1e-12).ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    let mut observed = clean.clone();
    let (good_sigma, bad_sigma) = (0.02 * rms, 0.20 * rms);
    for r in 0..n_rx {
        let sigma = if r % 4 == 0 { bad_sigma } else { good_sigma };
        for t in 0..nt {
            observed[[r, t]] += sigma * gauss();
        }
    }

    // Two inversions on identical data; only the weighting differs.
    let plain = FwiProcessor::new(parameters.clone());
    let pwls = FwiProcessor::new(parameters)
        .with_data_weighting(DataWeighting::InverseNoiseVariance { noise_window: 20 });

    let rec_plain = plain
        .invert_lbfgs(&observed, &initial, &geometry, &grid, 6)
        .expect("plain L2 inversion");
    let rec_pwls = pwls
        .invert_lbfgs(&observed, &initial, &geometry, &grid, 6)
        .expect("PWLS inversion");

    // Quantity of interest: mean absolute model error over the **anomaly core**
    // (voxels where the true anomaly contrast exceeds 20 m/s). Averaging over the
    // core rather than a single voxel makes the metric robust to the noise
    // realisation. This is the recovered-contrast accuracy PWLS is designed to
    // protect from the corrupting bad channels.
    let core_mae = |a: &Array3<f64>| -> f64 {
        let c0 = SOUND_SPEED_WATER_SIM;
        let (mut sum, mut n) = (0.0_f64, 0usize);
        for ([ix, iy, iz], &t) in truth.indexed_iter() {
            if t - c0 > 20.0 {
                sum += (a[[ix, iy, iz]] - t).abs();
                n += 1;
            }
        }
        sum / n as f64
    };
    let mae_plain = core_mae(&rec_plain);
    let mae_pwls = core_mae(&rec_pwls);
    assert!(
        mae_pwls < mae_plain,
        "PWLS must recover the anomaly core more accurately under bad-channel noise: \
         plain MAE = {mae_plain:.3}, PWLS MAE = {mae_pwls:.3}"
    );

    // Honesty note: PWLS protects the recovered contrast by down-weighting the
    // noisy channels, but down-weighting receivers shrinks the effective
    // aperture, so the *global* illuminated-region RMS is not guaranteed to fall —
    // a genuine bias/variance/aperture trade-off (the same one CT MBIR navigates).
    // We therefore assert only the quantity-of-interest accuracy above, not a
    // global RMS improvement.
}
