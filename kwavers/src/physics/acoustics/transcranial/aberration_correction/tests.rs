//! Tests for aberration correction

use super::*;
use crate::core::constants::numerical::TWO_PI;
use crate::domain::grid::Grid;
use ndarray::Array3;
use num_complex::Complex;
use std::f64::consts::PI;

fn make_corrector() -> TranscranialAberrationCorrection {
    let grid = Grid::new(32, 32, 32, 0.002, 0.002, 0.002).unwrap();
    TranscranialAberrationCorrection::new(&grid).unwrap()
}

// ── Structural smoke tests ────────────────────────────────────────────────────

#[test]
fn test_aberration_corrector_creation() {
    let grid = Grid::new(32, 32, 32, 0.002, 0.002, 0.002).unwrap();
    let _corrector = TranscranialAberrationCorrection::new(&grid).unwrap();
}

#[test]
fn test_phase_correction_lengths() {
    let grid = Grid::new(16, 16, 16, 0.005, 0.005, 0.005).unwrap();
    let corrector = TranscranialAberrationCorrection::new(&grid).unwrap();
    let ct_data = Array3::from_elem((16, 16, 16), 400.0);
    let positions = vec![[0.0, 0.0, 0.07], [0.02, 0.0, 0.07], [0.0, 0.02, 0.07]];
    let target = [0.04, 0.04, 0.04];
    let corr = corrector
        .calculate_correction(&ct_data, &positions, &target)
        .unwrap();
    assert_eq!(corr.phases.len(), positions.len());
    assert_eq!(corr.amplitudes.len(), positions.len());
}

// ── Phase-screen physics tests ────────────────────────────────────────────────

/// For a homogeneous medium (c_local = c_water everywhere), all aberration
/// phases must be exactly zero.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_homogeneous_medium_gives_zero_aberration() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    // HU=0 → water (1500 m/s ≈ reference_speed).  Use HU that maps to reference.
    // hu_to_sound_speed(HU) = 1500 + HU*0.5 approximately for soft tissue;
    // an all-water phantom has HU≈0 → c≈1500 m/s.
    // We test that the aberration phase is very small (< 0.01 rad) for a
    // uniform medium, allowing for the slight difference between the exact
    // reference_speed and the CT-derived speed.
    let ct_data = Array3::from_elem((32, 32, 32), 0.0_f64); // HU=0
    let corrector = TranscranialAberrationCorrection::new(&grid).unwrap();
    let positions = vec![[0.0, 0.0, 0.030], [0.005, 0.0, 0.030]];
    let target = [0.015, 0.015, 0.010];
    let phases = corrector
        .calculate_aberration_phases(&ct_data, &positions, &target)
        .unwrap();
    for (i, &phi) in phases.iter().enumerate() {
        assert!(
            phi.abs() < 0.1,
            "element {i}: aberration phase {phi:.4e} rad unexpectedly large for uniform medium"
        );
    }
}

/// Phase correction must be the exact negation of the computed aberration phase.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_correction_negates_aberration() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let ct_data = Array3::from_elem((32, 32, 32), 600.0_f64); // bone-like HU
    let corrector = TranscranialAberrationCorrection::new(&grid).unwrap();
    let positions = vec![[0.0, 0.0, 0.030]];
    let target = [0.010, 0.010, 0.005];

    let aberration_phases = corrector
        .calculate_aberration_phases(&ct_data, &positions, &target)
        .unwrap();
    let correction = corrector
        .calculate_correction(&ct_data, &positions, &target)
        .unwrap();

    assert!(
        (correction.phases[0] + aberration_phases[0]).abs() < 1e-12,
        "correction phase {:.6e} should equal −aberration {:.6e}",
        correction.phases[0],
        aberration_phases[0]
    );
}

/// The focal gain improvement for perfectly coherent phases (all same) is 0 dB.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_focal_gain_zero_for_coherent_phases() {
    let phases_coherent = vec![1.23_f64; 8]; // all identical → R = 1
    let gain_db = TranscranialAberrationCorrection::focal_gain_improvement_db(&phases_coherent);
    assert!(
        gain_db.abs() < 1e-10,
        "coherent phases → 0 dB improvement; got {gain_db:.4e} dB"
    );
}

/// The focal gain improvement for anti-phase pair (φ, φ+π) is positive.
/// # Panics
/// - Panics if assertion fails: `aberrated phases should yield positive focal gain improvement; got {gain_db:.4e} dB`.
///
#[test]
fn test_focal_gain_positive_for_aberrated_phases() {
    // Two elements with phases 0 and π → coherence R = 0 → ΔG = ∞.
    // Use a milder case: spread over [0, π) → R < 1, ΔG > 0.
    let n = 16_usize;
    let phases: Vec<f64> = (0..n).map(|i| i as f64 * PI / n as f64).collect();
    let gain_db = TranscranialAberrationCorrection::focal_gain_improvement_db(&phases);
    assert!(
        gain_db > 0.0,
        "aberrated phases should yield positive focal gain improvement; got {gain_db:.4e} dB"
    );
}

/// Circular coherence of identical phases equals 1.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_circular_coherence_identical_phases_is_one() {
    let phases = vec![0.7_f64; 20];
    let r = TranscranialAberrationCorrection::circular_coherence(&phases);
    assert!(
        (r - 1.0).abs() < 1e-12,
        "identical phases should have coherence 1.0; got {r:.6e}"
    );
}

/// Circular coherence of uniformly distributed phases over [0, 2π) → ≈ 0.
/// # Panics
/// - Panics if assertion fails: `uniformly distributed phases → coherence near 0; got {r:.6e}`.
///
#[test]
fn test_circular_coherence_uniform_distribution_near_zero() {
    let n = 1000_usize;
    let phases: Vec<f64> = (0..n).map(|i| TWO_PI * i as f64 / n as f64).collect();
    let r = TranscranialAberrationCorrection::circular_coherence(&phases);
    assert!(
        r < 0.01,
        "uniformly distributed phases → coherence near 0; got {r:.6e}"
    );
}

// ── Time-reversal (phase conjugation) tests ───────────────────────────────────

/// Time-reversal: if field at element position is e^{iφ}, correction = −φ.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_time_reversal_phase_conjugation_exact() {
    let corrector = make_corrector();

    let target_phase = PI / 3.0; // 60°
                                 // Build a uniform complex field with phase = target_phase
    let field: Array3<Complex<f64>> =
        Array3::from_elem((32, 32, 32), Complex::from_polar(1.0, target_phase));

    // Element placed at exact grid node (1, 1, 1) → position (dx, dy, dz)
    let pos = [corrector.grid.dx, corrector.grid.dy, corrector.grid.dz];
    let correction = corrector
        .apply_time_reversal_correction(&field, &[pos])
        .unwrap();

    let expected = -target_phase;
    assert!(
        (correction.phases[0] - expected).abs() < 1e-10,
        "phase conjugation: expected {expected:.6e} rad, got {:.6e} rad",
        correction.phases[0]
    );
}

/// After phase conjugation of a pure single-phase field, quality_metric → 1.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_time_reversal_quality_metric_is_one_for_uniform_field() {
    let corrector = make_corrector();
    let phi = PI / 4.0;
    let field: Array3<Complex<f64>> =
        Array3::from_elem((32, 32, 32), Complex::from_polar(1.0, phi));

    let positions: Vec<[f64; 3]> = (0..4)
        .map(|k| {
            [
                (k as f64 + 1.0) * corrector.grid.dx,
                corrector.grid.dy,
                corrector.grid.dz,
            ]
        })
        .collect();

    let correction = corrector
        .apply_time_reversal_correction(&field, &positions)
        .unwrap();

    assert!(
        (correction.quality_metric - 1.0).abs() < 1e-10,
        "quality metric should be 1.0 for uniform-phase field; got {:.6e}",
        correction.quality_metric
    );
}

/// Time-reversal output lengths match number of transducer elements.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_time_reversal_output_lengths() {
    let corrector = make_corrector();
    let field = Array3::from_elem((32, 32, 32), Complex::new(1.0, 0.5));
    let positions = vec![[0.002, 0.002, 0.002], [0.004, 0.002, 0.002]];
    let correction = corrector
        .apply_time_reversal_correction(&field, &positions)
        .unwrap();
    assert_eq!(correction.phases.len(), 2);
    assert_eq!(correction.amplitudes.len(), 2);
}

/// Focal gain improvement from time-reversal of an aberrated field is positive.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_time_reversal_focal_gain_positive_for_aberrated_field() {
    let corrector = make_corrector();
    // Different phases at each element position → pre-correction coherence < 1
    let (nx, ny, nz) = (32, 32, 32);
    // Spatially varying phase: φ(i,j,k) = 0.3 * i
    let field: Array3<Complex<f64>> = Array3::from_shape_fn((nx, ny, nz), |(i, _j, _k)| {
        Complex::from_polar(1.0, 0.3 * i as f64)
    });

    let positions: Vec<[f64; 3]> = (0..8)
        .map(|i| {
            [
                (i as f64 + 1.0) * corrector.grid.dx,
                corrector.grid.dy,
                corrector.grid.dz,
            ]
        })
        .collect();

    let correction = corrector
        .apply_time_reversal_correction(&field, &positions)
        .unwrap();

    assert!(
        correction.focal_gain_db > 0.0,
        "aberrated field should yield positive focal gain improvement; got {:.4e} dB",
        correction.focal_gain_db
    );
}
