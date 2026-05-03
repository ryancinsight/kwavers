use super::functions::{alignment_shifts_s, relative_delays_s};
use super::policy::DelayReference;

#[test]
fn sensor_index_reference_is_deterministic() {
    let delays = vec![0.010, 0.011, 0.009];
    let tau_ref = DelayReference::SensorIndex(0)
        .resolve_reference_delay_s(&delays)
        .expect("should resolve reference");
    assert!((tau_ref - 0.010).abs() < 1e-15);
}

#[test]
fn earliest_and_latest_reference() {
    let delays = vec![0.010, 0.011, 0.009];
    let tau_min = DelayReference::EarliestArrival
        .resolve_reference_delay_s(&delays)
        .expect("should resolve min");
    let tau_max = DelayReference::LatestArrival
        .resolve_reference_delay_s(&delays)
        .expect("should resolve max");
    assert!((tau_min - 0.009).abs() < 1e-15);
    assert!((tau_max - 0.011).abs() < 1e-15);
}

#[test]
fn relative_delays_match_definition() {
    let delays = vec![0.010, 0.011, 0.009];
    let rel = relative_delays_s(&delays, DelayReference::SensorIndex(0))
        .expect("should compute relative delays");
    assert_eq!(rel.len(), 3);
    assert!(
        (rel[0] - 0.0).abs() < 1e-15,
        "sensor 0 should have zero delay"
    );
    assert!(
        (rel[1] - 0.001).abs() < 1e-15,
        "sensor 1 should be 1ms later"
    );
    assert!(
        (rel[2] - (-0.001)).abs() < 1e-15,
        "sensor 2 should be 1ms earlier"
    );
}

#[test]
fn alignment_shifts_are_negative_relative_delays() {
    let delays = vec![0.010, 0.011, 0.009];
    let rel = relative_delays_s(&delays, DelayReference::SensorIndex(0)).expect("compute relative");
    let shifts =
        alignment_shifts_s(&delays, DelayReference::SensorIndex(0)).expect("compute alignment");
    for i in 0..delays.len() {
        assert!(
            (shifts[i] + rel[i]).abs() < 1e-15,
            "shift[{i}] should be -relative[{i}]"
        );
    }
}

#[test]
fn rejects_negative_delays() {
    let delays = vec![0.01, -0.01];
    let err = DelayReference::EarliestArrival
        .resolve_reference_delay_s(&delays)
        .expect_err("should reject negative delay");
    let msg = err.to_string();
    assert!(
        msg.contains("negative") || msg.contains("non-negative"),
        "error message should mention negative delays: {msg}"
    );
}

#[test]
fn rejects_non_finite_delays() {
    let delays = vec![0.01, f64::NAN];
    let err = DelayReference::EarliestArrival
        .resolve_reference_delay_s(&delays)
        .expect_err("should reject NaN");
    let msg = err.to_string();
    assert!(
        msg.contains("finite") || msg.contains("non-finite"),
        "error message should mention finite: {msg}"
    );

    let delays = vec![0.01, f64::INFINITY];
    let err = DelayReference::LatestArrival
        .resolve_reference_delay_s(&delays)
        .expect_err("should reject infinity");
    let msg = err.to_string();
    assert!(
        msg.contains("finite") || msg.contains("non-finite"),
        "error message should mention finite: {msg}"
    );
}

#[test]
fn rejects_out_of_bounds_sensor_index() {
    let delays = vec![0.01, 0.02];
    let err = DelayReference::SensorIndex(5)
        .resolve_reference_delay_s(&delays)
        .expect_err("should reject out-of-bounds index");
    assert!(
        err.to_string().contains("out of bounds"),
        "error should mention out of bounds"
    );
}

#[test]
fn rejects_empty_delays() {
    let delays: Vec<f64> = vec![];
    let err = DelayReference::SensorIndex(0)
        .resolve_reference_delay_s(&delays)
        .expect_err("should reject empty delays");
    assert!(
        err.to_string().contains("empty"),
        "error should mention empty"
    );
}

#[test]
fn recommended_default_is_sensor_zero() {
    assert_eq!(
        DelayReference::recommended_default(),
        DelayReference::SensorIndex(0)
    );
}

#[test]
fn relative_delays_can_be_negative() {
    let delays = vec![0.010, 0.011, 0.008];
    let rel = relative_delays_s(&delays, DelayReference::SensorIndex(0)).expect("compute relative");
    assert!(rel[2] < 0.0, "sensor 2 should have negative relative delay");
    assert!((rel[2] - (-0.002)).abs() < 1e-15);
}

#[test]
fn earliest_arrival_makes_all_relative_delays_non_negative() {
    let delays = vec![0.010, 0.011, 0.009];
    let rel =
        relative_delays_s(&delays, DelayReference::EarliestArrival).expect("compute relative");
    for (i, &r) in rel.iter().enumerate() {
        assert!(
            r >= -1e-14,
            "relative delay[{i}] = {r} should be non-negative with EarliestArrival"
        );
    }
}

#[test]
fn latest_arrival_makes_all_relative_delays_non_positive() {
    let delays = vec![0.010, 0.011, 0.009];
    let rel = relative_delays_s(&delays, DelayReference::LatestArrival).expect("compute relative");
    for (i, &r) in rel.iter().enumerate() {
        assert!(
            r <= 1e-14,
            "relative delay[{i}] = {r} should be non-positive with LatestArrival"
        );
    }
}
