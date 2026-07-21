use super::*;
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};

#[test]
fn mi_dimensional_check() {
    let mi = mechanical_index(-MPA_TO_PA, MHZ_TO_HZ);
    assert!((mi - 1.0).abs() < 1e-10);
}

#[test]
fn mi_rejects_nonpositive_frequency() {
    assert_eq!(mechanical_index(-MPA_TO_PA, 0.0), 0.0);
    assert_eq!(mechanical_index(-MPA_TO_PA, -MHZ_TO_HZ), 0.0);
}

#[test]
fn thermal_indices_are_nonnegative_ratios() {
    assert!((thermal_index_soft_tissue(210.0, 1.0) - 1.0).abs() < 1e-12);
    assert!((thermal_index_bone(40.0, 1.0) - 1.0).abs() < 1e-12);
}

#[test]
fn thermal_indices_reject_invalid_domains() {
    assert_eq!(thermal_index_soft_tissue(1.0, 0.0), 0.0);
    assert_eq!(thermal_index_soft_tissue(-1.0, 1.0), 0.0);
    assert_eq!(thermal_index_soft_tissue(f64::NAN, 1.0), 0.0);
    assert_eq!(thermal_index_bone(-1.0, 1.0), 0.0);
    assert_eq!(thermal_index_bone(1.0, -1.0), 0.0);
    assert_eq!(thermal_index_bone(1.0, f64::INFINITY), 0.0);
}

#[test]
fn thermal_index_cranial_matches_iec_definition() {
    assert!((thermal_index_cranial(80.0, 2.0) - 1.0).abs() < 1e-12);
    assert!((thermal_index_cranial(160.0, 2.0) - 2.0).abs() < 1e-12);
    assert!(thermal_index_cranial(80.0, 4.0) < thermal_index_cranial(80.0, 2.0));
    assert_eq!(thermal_index_cranial(-1.0, 2.0), 0.0);
    assert_eq!(thermal_index_cranial(80.0, 0.0), 0.0);
    assert_eq!(thermal_index_cranial(f64::NAN, 2.0), 0.0);
    assert_eq!(thermal_index_cranial(80.0, f64::INFINITY), 0.0);
}

#[test]
fn closed_loop_cem43_fixture_is_seeded_and_uses_cem43_contract() {
    let fixture = closed_loop_cem43_fixture(60, 0.5, 37.0, 60.0, 42).expect("valid CEM43 fixture");
    let repeated = closed_loop_cem43_fixture(60, 0.5, 37.0, 60.0, 42).expect("valid CEM43 fixture");
    let shifted_seed =
        closed_loop_cem43_fixture(60, 0.5, 37.0, 60.0, 43).expect("valid CEM43 fixture");

    assert_eq!(fixture, repeated);
    assert_ne!(
        fixture.feedback_temperature_c,
        shifted_seed.feedback_temperature_c
    );
    assert_eq!(fixture.time_s.len(), 60);
    assert_eq!(fixture.time_s[0], 0.0);
    assert_eq!(fixture.time_s[1], 0.5);
    assert_eq!(fixture.fixed_temperature_c[0], 37.0);
    assert_eq!(fixture.fixed_temperature_c[30], 60.0);
    assert_eq!(fixture.underdrive_temperature_c[40], 56.0);

    let expected_initial = 0.25_f64.powf(43.0 - 37.0) * 0.5 / 60.0;
    let bound = 8.0 * f64::EPSILON * expected_initial;
    assert!((fixture.fixed_cem43_min[0] - expected_initial).abs() <= bound);
    for dose in [
        &fixture.fixed_cem43_min,
        &fixture.feedback_cem43_min,
        &fixture.underdrive_cem43_min,
    ] {
        assert!(dose.windows(2).all(|w| w[1] >= w[0]));
    }
}

#[test]
fn closed_loop_cem43_fixture_rejects_invalid_inputs() {
    assert!(closed_loop_cem43_fixture(0, 0.5, 37.0, 60.0, 0).is_err());
    assert!(closed_loop_cem43_fixture(60, 0.0, 37.0, 60.0, 0).is_err());
    assert!(closed_loop_cem43_fixture(60, 0.5, f64::NAN, 60.0, 0).is_err());
    assert!(closed_loop_cem43_fixture(60, 0.5, 37.0, 37.0, 0).is_err());
}

#[test]
fn fda_limits_positive() {
    assert!(fda_ispta_limit_mw_cm2() > 0.0);
    assert!(fda_isppa_limit_w_cm2() > 0.0);
}

#[test]
fn mechanical_index_field_matches_scalar() {
    let pressures = [-MPA_TO_PA, -2.0 * MPA_TO_PA, -0.5 * MPA_TO_PA];
    let f_hz = 3.0 * MHZ_TO_HZ;
    let field = mechanical_index_field(&pressures, f_hz);
    for (&pressure, &field_value) in pressures.iter().zip(&field) {
        let scalar_value = mechanical_index(pressure, f_hz);
        assert!(
            (field_value - scalar_value).abs() < 1e-12,
            "field={field_value} scalar={scalar_value}"
        );
    }
}

#[test]
fn mechanical_index_frequency_sweep_matches_scalar() {
    let pressure = -0.45 * MPA_TO_PA;
    let frequencies = [0.25 * MHZ_TO_HZ, MHZ_TO_HZ, 2.25 * MHZ_TO_HZ];
    let sweep = mechanical_index_frequency_sweep(pressure, &frequencies);

    for (&frequency, &sweep_value) in frequencies.iter().zip(&sweep) {
        let scalar_value = mechanical_index(pressure, frequency);
        assert!(
            (sweep_value - scalar_value).abs() < 1e-12,
            "sweep={sweep_value} scalar={scalar_value}"
        );
    }
    assert!(sweep[0] > sweep[1]);
    assert!(sweep[1] > sweep[2]);
}

#[test]
fn mechanical_index_cavitation_risk_at_threshold_is_half() {
    let risk = mechanical_index_cavitation_risk(&[0.7], 0.7, 18.0);
    assert!(
        (risk[0] - 0.5).abs() < 1e-12,
        "P_risk(MI_thr)={} != 0.5",
        risk[0]
    );
}

#[test]
fn mechanical_index_cavitation_risk_matches_logistic_formula() {
    let mechanical_indices = [0.2, 0.7, 1.0, 1.9];
    let threshold = 0.7_f64;
    let slope = 18.0_f64;
    let risk = mechanical_index_cavitation_risk(&mechanical_indices, threshold, slope);
    for (actual, mechanical_index) in risk.iter().zip(mechanical_indices) {
        let expected = 1.0 / (1.0 + (-(slope * (mechanical_index - threshold))).exp());
        assert!(
            (actual - expected).abs() < 1e-12,
            "P_risk({mechanical_index})={actual} != {expected}"
        );
    }
}

#[test]
fn mechanical_index_cavitation_risk_is_bounded_and_monotone() {
    let mechanical_indices: Vec<f64> = (0..=20).map(|index| f64::from(index) * 0.1).collect();
    let risk = mechanical_index_cavitation_risk(&mechanical_indices, 0.7, 18.0);
    for (index, &value) in risk.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&value),
            "P_risk[{index}]={value} is outside [0, 1]"
        );
    }
    for index in 1..risk.len() {
        assert!(
            risk[index] >= risk[index - 1],
            "P_risk[{}]={} < P_risk[{}]={}",
            index,
            risk[index],
            index - 1,
            risk[index - 1]
        );
    }
}

#[test]
fn mechanical_index_cavitation_risk_rejects_invalid_domains() {
    assert_eq!(
        mechanical_index_cavitation_risk(&[0.7, f64::NAN], 0.7, 18.0),
        vec![0.5, 0.0]
    );
    assert_eq!(
        mechanical_index_cavitation_risk(&[0.7, 1.0], f64::NAN, 18.0),
        vec![0.0, 0.0]
    );
    assert_eq!(
        mechanical_index_cavitation_risk(&[0.7, 1.0], 0.7, 0.0),
        vec![0.0, 0.0]
    );
    assert_eq!(
        mechanical_index_cavitation_risk(&[0.7, 1.0], 0.7, f64::INFINITY),
        vec![0.0, 0.0]
    );
}
