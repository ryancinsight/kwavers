//! Unit tests for VelocityComponentStats.

use super::accumulator::VelocityComponentStats;
use leto::Array3;

// ── Theorem: max/min track signed extreme values ──────────────────────────────
//
// update() computes element-wise max/min (not |u|).  With two steps:
//   step 1: u = +1.0  → max = +1.0, min = +1.0
//   step 2: u = −2.0  → max = +1.0, min = −2.0
#[test]
fn test_max_min_track_signed_extremes() {
    let mut stats = VelocityComponentStats::new(4, 4, 1);

    stats.update(&Array3::from_elem([4, 4, 1], 1.0_f64));
    stats.update(&Array3::from_elem([4, 4, 1], -2.0_f64));

    assert_eq!(stats.global_max(), 1.0);
    assert_eq!(stats.global_min(), -2.0);
    assert_eq!(stats.time_step_count, 2);
}

// ── Theorem: RMS = sqrt( Σu² / N ) ───────────────────────────────────────────
//
// Two steps: u₁ = 3.0, u₂ = 4.0.
//   u_rms = sqrt((9 + 16) / 2) = sqrt(12.5)
#[test]
fn test_rms_two_steps() {
    let mut stats = VelocityComponentStats::new(2, 2, 1);

    stats.update(&Array3::from_elem([2, 2, 1], 3.0_f64));
    stats.update(&Array3::from_elem([2, 2, 1], 4.0_f64));

    let rms = stats.u_rms();
    let expected = ((9.0_f64 + 16.0) / 2.0).sqrt();
    assert!(
        (rms[[0, 0, 0]] - expected).abs() < 1e-12,
        "RMS: expected {expected}, got {}",
        rms[[0, 0, 0]]
    );
}

// ── Theorem: sample_max reads the correct voxel ───────────────────────────────
#[test]
fn test_sample_max_at_positions() {
    let mut stats = VelocityComponentStats::new(4, 4, 1);

    let mut u = Array3::zeros([4, 4, 1]);
    u[[2, 2, 0]] = 5.0;
    stats.update(&u);

    let positions = vec![(2, 2, 0), (0, 0, 0)];
    let max_vals = stats.sample_max(&positions);

    assert_eq!(max_vals[0], 5.0);
    assert!(max_vals[1] <= 0.0);
}

// ── Theorem: reset restores initial-state invariants ─────────────────────────
#[test]
fn test_reset_restores_initial_state() {
    let mut stats = VelocityComponentStats::new(2, 2, 1);
    stats.update(&Array3::from_elem([2, 2, 1], 3.0_f64));
    assert_eq!(stats.time_step_count, 1);

    stats.reset();
    assert_eq!(stats.time_step_count, 0);
    assert_eq!(stats.global_max(), f64::NEG_INFINITY);
    assert_eq!(stats.global_min(), f64::INFINITY);
    assert_eq!(stats.u_rms()[[0, 0, 0]], 0.0);
}

// ── Theorem: zero steps → u_rms = 0 ──────────────────────────────────────────
#[test]
fn test_u_rms_before_any_update_is_zero() {
    let stats = VelocityComponentStats::new(3, 3, 3);
    let rms = stats.u_rms();
    assert!(rms.iter().all(|&v| v == 0.0));
}

// ── Theorem: fill_rms matches sample_rms ─────────────────────────────────────
#[test]
fn test_fill_rms_matches_sample_rms() {
    use leto::Array1;

    let mut stats = VelocityComponentStats::new(4, 4, 1);
    stats.update(&Array3::from_elem([4, 4, 1], 2.0_f64));
    stats.update(&Array3::from_elem([4, 4, 1], 4.0_f64));

    let positions = vec![(0, 0, 0), (3, 3, 0)];

    let via_sample = stats.sample_rms(&positions);
    let mut via_fill = Array1::zeros([positions.len(]));
    stats.fill_rms(&positions, &mut via_fill).unwrap();

    for i in 0..positions.len() {
        assert!(
            (via_sample[i] - via_fill[i]).abs() < f64::EPSILON * 4.0,
            "sample vs fill RMS mismatch at sensor {i}"
        );
    }
}
