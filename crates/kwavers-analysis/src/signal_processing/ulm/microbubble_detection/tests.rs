use super::clutter::{svht_threshold, UlmSvdClutterFilter};
use super::localize::{gauss_newton_fit_2d, GaussianLocalizer};
use super::types::{GaussianLocalizationConfig, SvdClutterConfig};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_math::linear_algebra::LinearAlgebra;
use leto::Array2;

/// Generate a deterministic pseudo-noise matrix using a simple LCG for portability.
fn make_noise_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut state = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u1 = (state >> 11) as f64 / (1u64 << 53) as f64 + 1e-30;
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (TWO_PI * u2).cos()
    })
}

#[test]
fn test_svht_noise_only() {
    let n_px = 50usize;
    let n_t = 100usize;
    let noise = make_noise_matrix(n_px, n_t, 42);
    let (_u, sigma, _vt) = LinearAlgebra::svd(&noise).unwrap();

    let k = svht_threshold(&sigma, n_px, n_t);
    assert!(k <= 5, "SVHT on noise should give k≈0, got k={k}");
}

#[test]
fn test_svd_clutter_filter_rank1_tissue() {
    let n_px = 20;
    let n_t = 30;

    let tissue_spatial: Vec<f64> = (0..n_px).map(|i| (i as f64 * 0.1).sin()).collect();
    let tissue_temporal: Vec<f64> = (0..n_t).map(|t| (t as f64 * 0.05).cos() * 100.0).collect();

    let mut iq = Array2::<f64>::zeros((n_px, n_t));
    for i in 0..n_px {
        for t in 0..n_t {
            iq[[i, t]] = tissue_spatial[i] * tissue_temporal[t];
        }
    }
    for t in 5..8 {
        iq[[10, t]] += 5.0;
    }

    let cfg = SvdClutterConfig {
        fixed_clutter_rank: 1,
        ..Default::default()
    };
    let filter = UlmSvdClutterFilter::new(cfg);
    let (bubble, k) = filter.filter(&iq).unwrap();

    assert_eq!(k, 1, "Tissue rank should be 1");
    let bubble_energy: f64 = (5..8).map(|t| bubble[[10, t]].powi(2)).sum();
    let noise_energy: f64 = (15..20).map(|t| bubble[[10, t]].powi(2)).sum();
    assert!(
        bubble_energy > noise_energy * 5.0,
        "Bubble frames energy {bubble_energy:.3} should exceed background {noise_energy:.3}"
    );
}

#[test]
fn test_gaussian_fit_synthetic_peak() {
    let (nz, nx) = (11usize, 11usize);
    let (true_z, true_x) = (5.3_f64, 5.7_f64);
    let amp = 10.0_f64;
    let sigma = 1.2_f64;
    let bg = 0.5_f64;

    let mut envelope = Array2::<f64>::zeros((nz, nx));
    for iz in 0..nz {
        for ix in 0..nx {
            let dz = iz as f64 - true_z;
            let dx = ix as f64 - true_x;
            envelope[[iz, ix]] = amp * (-(dz * dz + dx * dx) / (2.0 * sigma * sigma)).exp() + bg;
        }
    }

    let result = gauss_newton_fit_2d(&envelope, true_z.round(), true_x.round(), 50);
    let (z_fit, x_fit, amp_fit, sigma_fit, _bg_fit) =
        result.expect("Gauss-Newton fit should converge");

    assert!(
        (z_fit - true_z).abs() < 0.1,
        "z center error {:.4} > 0.1 px",
        (z_fit - true_z).abs()
    );
    assert!(
        (x_fit - true_x).abs() < 0.1,
        "x center error {:.4} > 0.1 px",
        (x_fit - true_x).abs()
    );
    assert!(
        (amp_fit - amp).abs() / amp < 0.05,
        "Amplitude error {:.3}",
        (amp_fit - amp).abs() / amp
    );
    assert!(
        (sigma_fit - sigma).abs() / sigma < 0.05,
        "Sigma error {:.3}",
        (sigma_fit - sigma).abs() / sigma
    );
}

#[test]
fn test_localizer_detects_isolated_peak() {
    let (nz, nx) = (20, 20);
    let (tz, tx) = (10.4, 10.6);
    let mut envelope = Array2::<f64>::zeros((nz, nx));
    for iz in 0..nz {
        for ix in 0..nx {
            let dz = iz as f64 - tz;
            let dx = ix as f64 - tx;
            envelope[[iz, ix]] = 8.0 * (-(dz * dz + dx * dx) / 2.0).exp() + 0.1;
        }
    }

    let cfg = GaussianLocalizationConfig {
        min_sigma_px: 0.5,
        max_sigma_px: 3.0,
        ..Default::default()
    };
    let localizer = GaussianLocalizer::new(cfg);
    let detections = localizer.localize_frame(&envelope, 0).unwrap();

    assert_eq!(detections.len(), 1, "Should detect exactly one bubble");
    let d = &detections[0];
    assert!(
        (d.z - tz).abs() < 0.2,
        "z localization error {:.3}",
        (d.z - tz).abs()
    );
    assert!(
        (d.x - tx).abs() < 0.2,
        "x localization error {:.3}",
        (d.x - tx).abs()
    );
}
