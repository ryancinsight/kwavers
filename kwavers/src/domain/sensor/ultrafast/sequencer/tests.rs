use super::*;
use std::f64::consts::PI;

fn sequencer_40mm() -> TransmissionSequencer {
    TransmissionSequencer::new(1540.0, 0.040)
}

/// PRF_max = c / (2·z_max) = 1540 / 0.080 = 19 250 Hz.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_max_prf() {
    let seq = sequencer_40mm();
    let expected = 1540.0 / 0.080;
    assert!(
        (seq.max_prf() - expected).abs() / expected < 1e-10,
        "PRF_max mismatch: {:.1} Hz expected {:.1} Hz",
        seq.max_prf(),
        expected
    );
}

/// Setting PRF above PRF_max must return an error.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_prf_exceeds_max_returns_error() {
    let seq = sequencer_40mm();
    let prf_over = seq.max_prf() * 2.0;
    let result = seq.with_prf(prf_over);
    assert!(result.is_err(), "PRF exceeding max must return Err");
}

/// Setting PRF ≤ PRF_max must succeed.
/// # Panics
/// - Panics if `PRF = PRF_max must succeed`.
/// - Panics if `prf_override must be Some after with_prf`.
///
#[test]
fn test_prf_at_max_is_valid() {
    let seq = sequencer_40mm();
    let prf_max = seq.max_prf();
    let seq2 = seq.with_prf(prf_max).expect("PRF = PRF_max must succeed");
    // prf_override must be set to the exact requested value.
    let stored = seq2.prf_override.expect("prf_override must be Some after with_prf");
    assert!(
        (stored - prf_max).abs() < 1e-6,
        "prf_override = {stored} (expected {prf_max})"
    );
}

/// Sequential schedule has correct event count, timing, and PRF.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_sequential_schedule_timing() {
    let seq = sequencer_40mm();
    let angles: Vec<f64> = (-2..=2).map(|i| i as f64 * PI / 180.0).collect(); // 5 angles
    let sched = seq.sequential_schedule(&angles);

    assert_eq!(sched.n_events(), 5);
    let pri = 1.0 / seq.max_prf();
    for (k, ev) in sched.events.iter().enumerate() {
        let expected_t = k as f64 * pri;
        assert!(
            (ev.t_start - expected_t).abs() < 1e-15,
            "Event {k} t_start mismatch"
        );
    }
    assert!((sched.frame_rate - seq.max_prf() / 5.0).abs() < 1e-6);
}

/// Flash schedule has exactly one event at t=0 with θ=0.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_flash_schedule() {
    let seq = sequencer_40mm();
    let sched = seq.flash_schedule();
    assert_eq!(sched.n_events(), 1);
    assert_eq!(sched.events[0].event_index, 0);
    assert!(sched.events[0].t_start.abs() < 1e-15);
    assert!(sched.events[0].tilt_angle.abs() < 1e-15);
    assert!((sched.frame_rate - seq.max_prf()).abs() < 1e-6);
}

/// Interleaved schedule preserves all angles (as a multiset).
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_interleaved_schedule_all_angles_present() {
    let seq = sequencer_40mm();
    let angles: Vec<f64> = (-5..=5).map(|i| i as f64 * PI / 180.0).collect(); // 11 angles
    let sched = seq.interleaved_schedule(&angles);

    assert_eq!(sched.n_events(), 11);

    // Collect fired angles and sort
    let mut fired: Vec<i64> = sched
        .events
        .iter()
        .map(|ev| (ev.tilt_angle * 180.0 / PI).round() as i64)
        .collect();
    fired.sort();
    let mut expected: Vec<i64> = (-5..=5).collect();
    expected.sort();
    assert_eq!(
        fired, expected,
        "All 11 angles must appear in interleaved schedule"
    );
}

/// Interleaved schedule: first two events should have the maximum angular separation.
///
/// For 11 angles [-5°,…,5°] the interleaved order is [0, 5, 1, 6, …],
/// i.e., angles -5° and 0° at events 0 and 1.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_interleaved_max_separation_first_pair() {
    let seq = sequencer_40mm();
    let n = 11usize;
    let angles: Vec<f64> = (0..n).map(|i| i as f64 * PI / 180.0).collect(); // [0°, 1°, …, 10°]
    let sched = seq.interleaved_schedule(&angles);

    // Expected: event 0 → angle 0°, event 1 → angle 5° (half=5)
    let a0 = sched.events[0].tilt_angle;
    let a1 = sched.events[1].tilt_angle;
    let sep = (a1 - a0).abs();
    let expected_sep = 5.0 * PI / 180.0;
    assert!(
        (sep - expected_sep).abs() < 1e-12,
        "First pair separation {:.1}° ≠ 5°",
        sep * 180.0 / PI
    );
}

/// STA schedule: each event fires a different element, tilt_angle = 0.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_sta_schedule_element_ordering() {
    let seq = sequencer_40mm();
    let sched = seq.sta_schedule(8).unwrap();
    assert_eq!(sched.n_events(), 8);
    for (k, ev) in sched.events.iter().enumerate() {
        assert_eq!(ev.element_index, Some(k));
        assert!(ev.tilt_angle.abs() < 1e-15);
    }
}

/// STA schedule with 0 elements returns error.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_sta_zero_elements_errors() {
    let seq = sequencer_40mm();
    assert!(seq.sta_schedule(0).is_err());
}

/// Frame rate for N compounding events = PRF / N.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_frame_rate_formula() {
    let seq = sequencer_40mm();
    let n = 11usize;
    let expected = seq.max_prf() / n as f64;
    assert!(
        (seq.frame_rate(n) - expected).abs() / expected < 1e-10,
        "Frame rate {:.1} Hz ≠ {:.1} Hz",
        seq.frame_rate(n),
        expected
    );
}
