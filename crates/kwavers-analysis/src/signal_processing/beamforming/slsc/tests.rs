use super::beamformer::compute_lag_coherence;
use super::*;
use leto::{
    Array2,
    Array3,
};
use eunomia::Complex64;

#[test]
fn test_slsc_default_config() {
    let slsc = SlscBeamformer::new();
    assert_eq!(slsc.config().max_lag, 10);
    assert!(slsc.config().normalize);
}

#[test]
fn test_slsc_with_config() {
    let config = SlscConfig::with_max_lag(20);
    let slsc = SlscBeamformer::with_config(config);
    assert_eq!(slsc.config().max_lag, 20);
}

#[test]
fn test_slsc_process_simple() {
    let n_elements = 4;
    let n_samples = 10;
    let data = Array2::from_elem((n_elements, n_samples), Complex64::new(1.0, 0.0));

    let slsc = SlscBeamformer::new();
    let result = slsc.process(&data).expect("SLSC processing should succeed");

    assert_eq!(result.len(), n_samples);
    for &val in result.iter() {
        assert!((0.0..=1.0).contains(&val), "Coherence should be in [0, 1]");
    }
}

#[test]
fn test_slsc_rejects_single_element() {
    let data = Array2::from_elem((1, 10), Complex64::new(1.0, 0.0));
    let slsc = SlscBeamformer::new();
    let result = slsc.process(&data);
    assert!(result.is_err());
}

#[test]
fn test_lag_weighting_uniform() {
    let w = LagWeighting::Uniform;
    assert_eq!(w.weight(1, 10), 1.0);
    assert_eq!(w.weight(5, 10), 1.0);
}

#[test]
fn test_lag_weighting_triangular() {
    let w = LagWeighting::Triangular;
    assert_eq!(w.weight(0, 10), 1.0);
    assert_eq!(w.weight(5, 10), 0.5);
    assert_eq!(w.weight(10, 10), 0.0);
}

#[test]
fn test_slsc_grid_processing() {
    let n_elements = 8;
    let height = 10;
    let width = 20;
    let n_pixels = height * width;

    let data = Array2::from_elem((n_elements, n_pixels), Complex64::new(1.0, 0.0));
    let slsc = SlscBeamformer::new();
    let result = slsc
        .process_grid(&data, (height, width))
        .expect("Grid processing should succeed");

    assert_eq!(result.shape(), &[height, width]);
}

// ─── compute_lag_coherence exact value-semantic tests ────────────────────────

/// Identical signals at lag=1 give coherence = 1.0.
///
/// signals = [1+0j, 1+0j, 1+0j, 1+0j], lag=1.
/// For i in 0..3:
///   numerator += Re((1)·(1)*) = 1.0; energy1 += 1; energy2 += 1.
/// numerator = 3.0, energy1 = 3.0, energy2 = 3.0.
/// coherence = |3/√(3·3)| = |3/3| = 1.0.
#[test]
fn lag_coherence_identical_signals_is_one() {
    let signals = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let c = compute_lag_coherence(&signals, 1);
    assert!(
        (c - 1.0).abs() < 1e-14,
        "identical-signals coherence at lag=1 = {c} (expected 1.0)"
    );
}

/// Orthogonal signals at lag=1 give coherence = 0.0.
///
/// signals = [1+0j, 0+1j], lag=1.
/// i=0: Re((1+0j)·(0-1j)) = Re(-j) = 0. numerator = 0.
/// energy1 = 1, energy2 = 1 → denominator = 1 → coherence = 0.
#[test]
fn lag_coherence_orthogonal_signals_is_zero() {
    let signals = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
    let c = compute_lag_coherence(&signals, 1);
    assert!(
        c.abs() < 1e-14,
        "orthogonal-signals coherence at lag=1 = {c} (expected 0.0)"
    );
}

/// Anti-phase signals at lag=1 give coherence = 1.0 (abs absorbs sign).
///
/// signals = [1+0j, -1+0j], lag=1.
/// i=0: Re((1+0j)·(-1+0j)*) = Re((1+0j)·(-1-0j)) = Re(-1) = -1.
/// energy1=1, energy2=1 → denominator=1 → |(-1)/1| = 1.0.
#[test]
fn lag_coherence_anti_phase_signals_is_one() {
    let signals = vec![Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
    let c = compute_lag_coherence(&signals, 1);
    assert!(
        (c - 1.0).abs() < 1e-14,
        "anti-phase coherence at lag=1 = {c} (expected 1.0, abs captures sign)"
    );
}

/// Zero-energy signal at lag=1 returns 0.0 (guard clause).
///
/// signals = [1+0j, 0+0j], lag=1.
/// energy2 = 0 → denominator = 0 < 1e-10 → return 0.0.
#[test]
fn lag_coherence_zero_energy_signal_returns_zero() {
    let signals = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let c = compute_lag_coherence(&signals, 1);
    assert!(
        c.abs() < 1e-14,
        "zero-energy coherence = {c} (expected 0.0)"
    );
}

/// `compute_lag_coherence` with lag ≥ n returns 0.0 (out-of-bounds guard).
///
/// signals of length 3, lag=3: lag >= n → early return 0.0.
#[test]
fn lag_coherence_lag_exceeds_length_returns_zero() {
    let signals = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let c = compute_lag_coherence(&signals, 3); // lag=3 >= n=3
    assert!(c.abs() < 1e-14, "lag≥n coherence = {c} (expected 0.0)");
}

/// `process` on identical-signal input yields all-ones coherence vector.
///
/// All elements of data = 1+0j → coherence at every sample = 1.0.
/// Uses max_lag = min(10, n_elements-1) = 3 (n_elements=4).
#[test]
fn slsc_process_identical_input_yields_all_ones() {
    let data = Array2::from_elem((4, 8), Complex64::new(1.0, 0.0));
    let slsc = SlscBeamformer::new();
    let result = slsc.process(&data).unwrap();
    for (i, &c) in result.iter().enumerate() {
        assert!(
            (c - 1.0).abs() < 1e-12,
            "coherence[{i}] = {c} (expected 1.0 for identical signals)"
        );
    }
}

#[test]
fn slsc_process_parallel_identical_input_yields_all_ones() {
    let data = Array2::from_elem((4, 8), Complex64::new(1.0, 0.0));
    let slsc = SlscBeamformer::new();

    let result = slsc
        .process_parallel(&data)
        .expect("parallel SLSC processing should succeed");

    assert_eq!(result.len(), 8);
    for (i, &c) in result.iter().enumerate() {
        assert!(
            (c - 1.0).abs() < 1e-12,
            "parallel coherence[{i}] = {c} (expected 1.0 for identical signals)"
        );
    }
}

#[test]
fn slsc_process_volume_identical_input_yields_all_ones() {
    let data = Array3::from_elem((4, 3, 5), Complex64::new(1.0, 0.0));
    let slsc = SlscBeamformer::new();

    let result = slsc
        .process_volume(&data)
        .expect("volume SLSC processing should succeed");

    assert_eq!(result.shape(), &[3, 5]);
    for beam in 0..3 {
        for sample in 0..5 {
            let c = result[[beam, sample]];
            assert!(
                (c - 1.0).abs() < 1e-12,
                "volume coherence[{beam},{sample}] = {c} (expected 1.0)"
            );
        }
    }
}

#[test]
fn slsc_coherence_map_clamps_to_unit_interval() {
    let data = Array2::from_elem((4, 4), Complex64::new(1.0, 0.0));
    let slsc = SlscBeamformer::new();

    let map = slsc
        .create_coherence_map(&data, Some(40.0))
        .expect("coherence map should succeed");

    assert_eq!(map.shape(), &[1, 4]);
    for (idx, &value) in map.iter().enumerate() {
        assert_eq!(value, 1.0, "map[{idx}] should be clamped to 1.0");
    }
}

/// `LagWeighting::Triangular` at lag=max_lag/2 is exactly 0.5.
///
/// weight(5, 10) = 1 - 5/10 = 0.5 (exact IEEE 754).
#[test]
fn triangular_weighting_midpoint_is_half() {
    let w = LagWeighting::Triangular;
    let weight = w.weight(5, 10);
    assert!(
        (weight - 0.5).abs() < 1e-15,
        "Triangular weight(5,10) = {weight} (expected 0.5)"
    );
}

/// `LagWeighting::Hamming` at lag=0 equals α = 0.54.
///
/// Hamming formula: α - β·cos(2π·0/max_lag) = 0.54 - 0.46·cos(0) = 0.54 - 0.46 = 0.08.
///
/// Note: lag=0 → coefficient = 0.54 - 0.46·1 = 0.08.
#[test]
fn hamming_weighting_lag_zero_is_point_zero_eight() {
    let w = LagWeighting::Hamming;
    let weight = w.weight(0, 10);
    assert!(
        (weight - 0.08).abs() < 1e-14,
        "Hamming weight(0,10) = {weight} (expected 0.08 = 0.54-0.46)"
    );
}

// ─── process_slsc_batch exact value tests ─────────────────────────────────────

/// `process_slsc_batch` on all-identical frames of identical signals yields all-ones output.
///
/// input shape: (4 elements, 2 frames, 8 samples), all = 1+0j.
/// Each frame is a 4×8 all-ones matrix → SLSC coherence = 1.0 per sample.
/// Output shape: (2, 8) = [[1.0, …], [1.0, …]].
#[test]
fn slsc_batch_identical_input_yields_all_ones() {
    let n_elements = 4;
    let n_frames = 2;
    let n_samples = 8;
    let data = Array3::from_elem((n_elements, n_frames, n_samples), Complex64::new(1.0, 0.0));
    let config = SlscConfig::default();
    let result = process_slsc_batch(&data, &config).expect("batch processing should succeed");

    assert_eq!(
        result.shape(),
        &[n_frames, n_samples],
        "output shape must be (n_frames, n_samples)"
    );
    for frame in 0..n_frames {
        for s in 0..n_samples {
            let c = result[[frame, s]];
            assert!(
                (c - 1.0).abs() < 1e-12,
                "batch result[{frame},{s}] = {c} (expected 1.0 for identical signals)"
            );
        }
    }
}

/// `process_slsc_batch` rejects input with n_elements < 2.
///
/// shape (1, 3, 8): n_elements=1 is below the 2-element minimum.
#[test]
fn slsc_batch_rejects_single_element() {
    let data = Array3::from_elem((1, 3, 8), Complex64::new(1.0, 0.0));
    let config = SlscConfig::default();
    let result = process_slsc_batch(&data, &config);
    assert!(result.is_err(), "batch with n_elements=1 must return Err");
}

// ─── MultiLagSlsc exact value tests ───────────────────────────────────────────

/// `MultiLagSlsc` with two configs and uniform weights (0.5+0.5) on identical
/// signals gives all-ones output.
///
/// Each config outputs coherence=1.0 per sample; combined = 0.5·1 + 0.5·1 = 1.0.
#[test]
fn multi_lag_slsc_identical_signals_yields_all_ones() {
    let configs = vec![SlscConfig::with_max_lag(3), SlscConfig::with_max_lag(5)];
    let multi = MultiLagSlsc::with_configs(configs);
    let data = Array2::from_elem((4, 8), Complex64::new(1.0, 0.0));

    let result = multi
        .process_multi(&data)
        .expect("multi-lag processing should succeed");

    assert_eq!(result.len(), 8, "output length must equal n_samples");
    for (i, &c) in result.iter().enumerate() {
        assert!(
            (c - 1.0).abs() < 1e-12,
            "multi_lag result[{i}] = {c} (expected 1.0)"
        );
    }
}

/// `MultiLagSlsc` with a single config and weight 1.0 is equivalent to direct SLSC.
///
/// Single-config multi-lag with weight=1/1=1.0 must match single SlscBeamformer output.
#[test]
fn multi_lag_slsc_single_config_matches_direct_slsc() {
    let config = SlscConfig::with_max_lag(4);
    let multi = MultiLagSlsc::with_configs(vec![config.clone()]);
    let slsc = SlscBeamformer::with_config(config);
    let data = Array2::from_elem((6, 10), Complex64::new(1.0, 0.0));

    let multi_result = multi.process_multi(&data).unwrap();
    let direct_result = slsc.process(&data).unwrap();

    assert_eq!(
        multi_result.len(),
        direct_result.len(),
        "lengths must match"
    );
    for (i, (&m, &d)) in multi_result.iter().zip(direct_result.iter()).enumerate() {
        assert!(
            (m - d).abs() < 1e-14,
            "multi_lag[{i}]={m} vs direct[{i}]={d} — must be identical"
        );
    }
}

// ─── AdaptiveSlsc exact value tests ───────────────────────────────────────────

/// `AdaptiveSlsc::process_adaptive` on identical signals yields all-ones coherence.
///
/// For identical data every lag has coherence=1.0 so the chosen lag doesn't
/// affect the output; the final coherence vector must be all 1.0.
#[test]
fn adaptive_slsc_identical_signals_yields_all_ones() {
    let data = Array2::from_elem((4, 8), Complex64::new(1.0, 0.0));
    let adaptive = AdaptiveSlsc::new();
    let result = adaptive
        .process_adaptive(&data)
        .expect("adaptive SLSC should succeed");

    assert_eq!(result.len(), 8, "output length must equal n_samples");
    for (i, &c) in result.iter().enumerate() {
        assert!(
            (c - 1.0).abs() < 1e-12,
            "adaptive_slsc result[{i}] = {c} (expected 1.0 for identical signals)"
        );
    }
}

/// `AdaptiveSlsc` output is in [0, 1] for arbitrary bounded signals.
///
/// Uses a linearly varying signal per element to exercise the adaptation path
/// without identical signals; coherence must remain in [0, 1].
#[test]
fn adaptive_slsc_output_bounded_in_zero_one() {
    let n_elements = 6;
    let n_samples = 12;
    let mut data = Array2::zeros((n_elements, n_samples));
    for i in 0..n_elements {
        for s in 0..n_samples {
            let phase = (i as f64 + s as f64) * 0.3;
            data[[i, s]] = Complex64::new(phase.cos(), phase.sin());
        }
    }
    let adaptive = AdaptiveSlsc::new();
    let result = adaptive
        .process_adaptive(&data)
        .expect("adaptive SLSC should succeed");
    for (i, &c) in result.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&c),
            "adaptive_slsc result[{i}] = {c} out of [0,1]"
        );
    }
}

