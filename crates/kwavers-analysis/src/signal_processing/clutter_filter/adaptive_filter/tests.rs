use super::filter::AdaptiveFilter;
use super::types::{AdaptiveFilterConfig, CbrEstimationMethod, SubspaceSeparationMethod};
use kwavers_core::constants::numerical::TWO_PI;
use leto::Array2;

#[test]
fn test_adaptive_filter_creation() {
    let config = AdaptiveFilterConfig::default();
    let _filter = AdaptiveFilter::new(config).unwrap();
}

#[test]
fn test_config_validation() {
    let config = AdaptiveFilterConfig {
        noise_floor_threshold: 1.5,
        ..Default::default()
    };
    assert!(AdaptiveFilter::new(config).is_err());

    let config = AdaptiveFilterConfig {
        temporal_smoothing: true,
        smoothing_window: 0,
        ..Default::default()
    };
    assert!(AdaptiveFilter::new(config).is_err());

    let config = AdaptiveFilterConfig {
        separation_method: SubspaceSeparationMethod::FixedRank { clutter_rank: 0 },
        ..Default::default()
    };
    assert!(AdaptiveFilter::new(config).is_err());
}

#[test]
fn test_filter_removes_low_frequency_component() {
    let config = AdaptiveFilterConfig {
        separation_method: SubspaceSeparationMethod::FixedRank { clutter_rank: 1 },
        cbr_estimation: CbrEstimationMethod::EigenvalueSum,
        noise_floor_threshold: 1e-6,
        temporal_smoothing: false,
        smoothing_window: 1,
    };

    let mut filter = AdaptiveFilter::new(config).unwrap();

    let n_frames = 16;
    let dc_component = 10.0;
    let mut data = Array2::<f64>::zeros((1, n_frames));
    for t in 0..n_frames {
        let oscillation = (TWO_PI * t as f64 / 4.0).cos();
        data[[0, t]] = dc_component + oscillation;
    }

    let filtered = filter.filter(&data).unwrap();
    let filtered_mean: f64 =
        leto::mean_all(&filtered.index_axis::<1>(0, 0).unwrap().to_contiguous()).unwrap();
    assert!(filtered_mean.abs() < 0.5 * dc_component);
}

#[test]
fn test_adaptive_threshold_method() {
    let config = AdaptiveFilterConfig {
        separation_method: SubspaceSeparationMethod::AdaptiveThreshold { decay_factor: 0.2 },
        cbr_estimation: CbrEstimationMethod::EigenvalueSum,
        noise_floor_threshold: 1e-6,
        temporal_smoothing: false,
        smoothing_window: 1,
    };

    let mut filter = AdaptiveFilter::new(config).unwrap();

    let n_frames = 16;
    let mut data = Array2::<f64>::zeros((1, n_frames));
    for t in 0..n_frames {
        let low_freq = 5.0 * (TWO_PI * t as f64 / 16.0).cos();
        let high_freq = 1.0 * (TWO_PI * t as f64 / 2.0).cos();
        data[[0, t]] = low_freq + high_freq;
    }

    let filtered = filter.filter(&data).unwrap();
    assert!(filtered.iter().all(|&x| x.is_finite()));
    let cbr = filter.current_cbr_db().unwrap();
    assert!(cbr.is_finite());
}

#[test]
fn test_cbr_based_separation() {
    let config = AdaptiveFilterConfig {
        separation_method: SubspaceSeparationMethod::CbrBased {
            target_cbr_db: 20.0,
        },
        cbr_estimation: CbrEstimationMethod::PowerRatio,
        noise_floor_threshold: 1e-6,
        temporal_smoothing: false,
        smoothing_window: 1,
    };

    let mut filter = AdaptiveFilter::new(config).unwrap();

    let n_frames = 32;
    let mut data = Array2::<f64>::zeros((1, n_frames));
    for t in 0..n_frames {
        let clutter = 10.0 * (TWO_PI * t as f64 / 32.0).sin();
        let blood = 0.5 * (TWO_PI * t as f64 / 4.0).sin();
        data[[0, t]] = clutter + blood;
    }

    let _filtered = filter.filter(&data).unwrap();
    let cbr_db = filter.current_cbr_db().unwrap();
    assert!(cbr_db.is_finite());
    assert!(cbr_db > 0.0);
}

#[test]
fn test_filter_preserves_high_frequency() {
    let config = AdaptiveFilterConfig {
        separation_method: SubspaceSeparationMethod::FixedRank { clutter_rank: 2 },
        cbr_estimation: CbrEstimationMethod::EigenvalueSum,
        noise_floor_threshold: 1e-6,
        temporal_smoothing: false,
        smoothing_window: 1,
    };

    let mut filter = AdaptiveFilter::new(config).unwrap();

    let n_frames = 16;
    let mut data = Array2::<f64>::zeros((1, n_frames));
    for t in 0..n_frames {
        data[[0, t]] = (TWO_PI * t as f64 / 2.0).sin();
    }

    let original_power: f64 = data.iter().map(|&x| x * x).sum();
    let filtered = filter.filter(&data).unwrap();
    let filtered_power: f64 = filtered.iter().map(|&x| x * x).sum();
    assert!(filtered_power > 0.3 * original_power);
}

#[test]
fn test_insufficient_frames() {
    let config = AdaptiveFilterConfig::default();
    let mut filter = AdaptiveFilter::new(config).unwrap();
    let data = Array2::<f64>::zeros((1, 2));
    assert!(filter.filter(&data).is_err());
}

#[test]
fn test_cbr_history() {
    let config = AdaptiveFilterConfig::default();
    let mut filter = AdaptiveFilter::new(config).unwrap();

    let data = Array2::<f64>::from_shape_fn((3, 16), |[_, t]| (TWO_PI * t as f64 / 8.0).sin());

    filter.filter(&data).unwrap();
    assert_eq!(filter.cbr_history().len(), 3);

    filter.clear_history();
    assert_eq!(filter.cbr_history().len(), 0);
}

/// Per-pixel result of the classical-Jacobi filter, the eigensolver the
/// adaptive filter used before it moved to tridiagonal QL.
struct JacobiReference {
    filtered: Vec<f64>,
    /// Eigenvalues, descending.
    eigenvalues: Vec<f64>,
    frobenius_norm: f64,
    rank: usize,
}

fn jacobi_reference(signal: &[f64], rank_of: impl Fn(&[f64]) -> usize) -> JacobiReference {
    let n = signal.len();
    let lag_mean = |lag: usize| -> f64 {
        let count = n - lag;
        (0..count).map(|t| signal[t] * signal[t + lag]).sum::<f64>() / count_f64(count)
    };
    let covariance = Array2::from_shape_fn((n, n), |[i, j]| lag_mean(i.abs_diff(j)));
    let frobenius_norm = covariance.iter().map(|v| v * v).sum::<f64>().sqrt();
    let decomposition =
        leto_ops::symmetric_eigen_jacobi(&covariance.view()).expect("invariant: symmetric input");
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| decomposition.eigenvalues[b].total_cmp(&decomposition.eigenvalues[a]));
    let eigenvalues: Vec<f64> = order
        .iter()
        .map(|&k| decomposition.eigenvalues[k])
        .collect();
    let rank = rank_of(&eigenvalues);
    let mut filtered = signal.to_vec();
    for &column in order.iter().take(rank) {
        let eigenvector = |row: usize| decomposition.eigenvectors[[row, column]];
        let coefficient: f64 = (0..n).map(|row| signal[row] * eigenvector(row)).sum();
        for (row, value) in filtered.iter_mut().enumerate() {
            *value -= coefficient * eigenvector(row);
        }
    }
    JacobiReference {
        filtered,
        eigenvalues,
        frobenius_norm,
        rank,
    }
}

fn count_f64(count: usize) -> f64 {
    f64::from(u32::try_from(count).expect("invariant: counts in this module fit in u32"))
}

/// The tridiagonal-QL filter agrees with the Jacobi filter on the
/// `generate_fus_data(30, 120, 5.0, 0.5, 0.02, 0.15)` ensemble of the
/// integration suite (tissue amplitude 5 at 0.02 cycles/frame, blood amplitude
/// `0.5 (1 + i/10)` at 0.15), at pixels 0 and 29 (one Jacobi decomposition of
/// a 120 x 120 covariance costs a few hundred milliseconds).
///
/// Both solvers return the exact eigendecomposition of `R + E` with
/// `||E||_2 <= n^2 eps ||R||_F` (the envelope `leto-ops` documents for QL;
/// Jacobi's is no larger), so the two covariances differ by at most
/// `2 n^2 eps ||R||_F =: e`. The rank-`k` eigenspace of two matrices that far
/// apart differs by `sin(theta) <= e / (gap - e)` (Davis-Kahan), `gap =
/// lambda_k - lambda_(k+1)`, hence
/// `||f_QL - f_Jacobi||_2 <= ||x||_2 e / (gap - e)`. The two pixels have large
/// gaps and select different adaptive ranks (2 at pixel 0, 4 at pixel 29).
fn assert_filter_matches_jacobi(
    separation_method: SubspaceSeparationMethod,
    rank_of: impl Fn(&[f64]) -> usize,
    expected_ranks: [usize; 2],
) {
    const N_FRAMES: usize = 120;
    const PIXELS: [usize; 2] = [0, 29];
    let data = Array2::from_shape_fn((PIXELS.len(), N_FRAMES), |[row, frame]| {
        let t = count_f64(frame);
        let blood_scale = count_f64(PIXELS[row]) / 10.0 + 1.0;
        5.0 * (TWO_PI * 0.02 * t).sin() + 0.5 * blood_scale * (TWO_PI * 0.15 * t).sin()
    });
    let mut filter = AdaptiveFilter::new(AdaptiveFilterConfig {
        separation_method,
        ..Default::default()
    })
    .unwrap();
    let filtered = filter.filter(&data).unwrap();

    let n = count_f64(N_FRAMES);
    for (row, expected_rank) in expected_ranks.into_iter().enumerate() {
        let signal: Vec<f64> = (0..N_FRAMES).map(|t| data[[row, t]]).collect();
        let reference = jacobi_reference(&signal, &rank_of);
        assert_eq!(
            reference.rank, expected_rank,
            "pixel {}: fixture rank",
            PIXELS[row]
        );

        let backward_error = 2.0 * n * n * f64::EPSILON * reference.frobenius_norm;
        let gap = reference.eigenvalues[reference.rank - 1] - reference.eigenvalues[reference.rank];
        assert!(
            gap > 10.0 * backward_error,
            "pixel {}: gap {gap} too small",
            PIXELS[row]
        );
        let signal_norm = signal.iter().map(|v| v * v).sum::<f64>().sqrt();
        let tolerance = signal_norm * backward_error / (gap - backward_error);

        let difference = (0..N_FRAMES)
            .map(|t| (filtered[[row, t]] - reference.filtered[t]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            difference <= tolerance,
            "pixel {}: |f_QL - f_Jacobi| = {difference} exceeds {tolerance}",
            PIXELS[row]
        );
        let removed = signal
            .iter()
            .zip(&reference.filtered)
            .map(|(x, f)| (x - f).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            removed > 0.5 * signal_norm,
            "pixel {}: reference removes the tissue",
            PIXELS[row]
        );
    }
}

#[test]
fn test_fixed_rank_matches_jacobi_reference() {
    assert_filter_matches_jacobi(
        SubspaceSeparationMethod::FixedRank { clutter_rank: 2 },
        |_| 2,
        [2, 2],
    );
}

#[test]
fn test_adaptive_threshold_matches_jacobi_reference() {
    // Default noise floor 1e-6; rank = first eigenvalue below
    // `0.1 lambda_max` or the floor, as `determine_clutter_rank` defines it.
    assert_filter_matches_jacobi(
        SubspaceSeparationMethod::AdaptiveThreshold { decay_factor: 0.1 },
        |eigenvalues| {
            let largest = eigenvalues[0];
            eigenvalues
                .iter()
                .position(|&e| e < 0.1 * largest || e < 1e-6 * largest)
                .unwrap_or(eigenvalues.len() / 2)
                .max(1)
        },
        [2, 4],
    );
}
