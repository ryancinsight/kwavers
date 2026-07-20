use super::*;
use crate::ml::uncertainty::conformal_prediction::config::ConformalConfig;
use leto::Array2;

#[test]
fn test_conformal_predictor_creation() {
    let config = ConformalConfig {
        confidence_level: 0.9,
        calibration_size: 2,
    };
    let predictor = MlConformalPredictor::new(config).unwrap();
    assert!(
        !predictor.is_calibrated(),
        "fresh predictor must not be calibrated"
    );
    assert!(
        predictor.calibration_scores.is_empty(),
        "calibration_scores must be empty before calibrate()"
    );
}

#[test]
fn test_conformal_predictor_rejects_invalid_confidence() {
    let err = MlConformalPredictor::new(ConformalConfig {
        confidence_level: 0.0,
        calibration_size: 100,
    })
    .unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("Confidence level"),
        "zero confidence_level error must mention 'Confidence level'; got: {msg}"
    );

    let err2 = MlConformalPredictor::new(ConformalConfig {
        confidence_level: 1.0,
        calibration_size: 100,
    })
    .unwrap_err();
    let msg2 = format!("{err2:?}");
    assert!(
        msg2.contains("Confidence level"),
        "unit confidence_level error must mention 'Confidence level'; got: {msg2}"
    );
}

#[test]
fn test_conformal_calibration() {
    let mut predictor = MlConformalPredictor::new(ConformalConfig {
        confidence_level: 0.9,
        calibration_size: 2,
    })
    .unwrap();

    let predictions = vec![
        Array2::from_elem((5, 5), 1.0_f32),
        Array2::from_elem((5, 5), 2.0_f32),
    ];
    let targets = vec![
        Array2::from_elem((5, 5), 1.1_f32),
        Array2::from_elem((5, 5), 1.9_f32),
    ];

    predictor.calibrate(&predictions, &targets).unwrap();
    assert!(
        predictor.is_calibrated(),
        "predictor must be calibrated after calibrate()"
    );
    assert_eq!(
        predictor.calibration_scores.len(),
        2,
        "must store one score per calibration pair"
    );
    for (i, &score) in predictor.calibration_scores.iter().enumerate() {
        let err = (score - 0.1).abs();
        assert!(
            err < 1e-5,
            "calibration_scores[{i}] = {score} (expected 0.1)"
        );
    }
}

#[test]
fn test_conformity_score_computation() {
    let predictor = MlConformalPredictor::new(ConformalConfig::default()).unwrap();
    let prediction = Array2::from_elem((3, 3), 1.0_f32);
    let target = Array2::from_elem((3, 3), 1.2_f32);

    let score = predictor
        .compute_conformity_score(&prediction, &target)
        .unwrap();
    let err = (score - 0.2).abs();
    assert!(
        err < 1e-5,
        "conformity score = {score} (expected 0.2 for uniform offset 0.2)"
    );
    assert!(
        score > 0.0,
        "conformity score must be positive for non-matching prediction"
    );
}

#[test]
fn even_conformity_median_preserves_prediction_precision() {
    let predictor = MlConformalPredictor::new(ConformalConfig::default()).unwrap();
    let lower = 1.0_f32;
    let upper = f32::from_bits(lower.to_bits() + 1);
    let prediction = Array2::from_elem((1, 2), 0.0_f32);
    let target = Array2::from_shape_vec((1, 2), vec![lower, upper]).unwrap();

    let score = predictor
        .compute_conformity_score(&prediction, &target)
        .unwrap();
    let expected = (lower + upper) / 2.0_f32;

    assert_eq!(
        score.to_bits(),
        f64::from(expected).to_bits(),
        "the conformity reduction must execute in the prediction precision"
    );
}

#[test]
fn test_quantile_computation() {
    let mut predictor = MlConformalPredictor::new(ConformalConfig::default()).unwrap();
    predictor.calibration_scores = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    predictor.is_calibrated = true;

    let quantile = predictor.compute_quantile(0.9).unwrap();
    assert_eq!(
        quantile.to_bits(),
        0.5_f64.to_bits(),
        "the finite-sample 90% rank for five scores is capped at the fifth score"
    );
}

#[test]
fn test_empty_calibration_and_shape_mismatch_are_rejected() {
    let mut predictor = MlConformalPredictor::new(ConformalConfig {
        confidence_level: 0.9,
        calibration_size: 1,
    })
    .unwrap();
    let empty_error = predictor.calibrate(&[], &[]).unwrap_err();
    assert!(
        format!("{empty_error:?}").contains("requires 1"),
        "empty calibration must report the configured cardinality"
    );

    let prediction = Array2::from_elem((1, 2), 1.0_f32);
    let target = Array2::from_elem((2, 1), 1.0_f32);
    let shape_error = predictor.calibrate(&[prediction], &[target]).unwrap_err();
    assert!(
        format!("{shape_error:?}").contains("does not match target shape"),
        "shape error must identify both array shapes"
    );
    assert!(
        !predictor.is_calibrated(),
        "a failed calibration must not mutate predictor state"
    );
}

#[test]
fn non_finite_calibration_and_prediction_inputs_are_rejected() {
    let mut predictor = MlConformalPredictor::new(ConformalConfig {
        confidence_level: 0.9,
        calibration_size: 1,
    })
    .unwrap();
    let target = Array2::from_elem((1, 1), 1.0_f32);
    let non_finite = Array2::from_elem((1, 1), f32::NAN);
    let calibration_error = predictor
        .calibrate(
            std::slice::from_ref(&non_finite),
            std::slice::from_ref(&target),
        )
        .unwrap_err();
    assert!(
        format!("{calibration_error:?}").contains("non-finite value at element 0"),
        "non-finite calibration error must identify the invalid element"
    );
    assert!(!predictor.is_calibrated());

    predictor
        .calibrate(std::slice::from_ref(&target), std::slice::from_ref(&target))
        .unwrap();
    let prediction_error = predictor
        .predict_intervals(std::slice::from_ref(&non_finite))
        .unwrap_err();
    assert!(
        format!("{prediction_error:?}").contains("Prediction sample 0"),
        "non-finite interval input must identify its sample"
    );
}

#[test]
fn test_prediction_intervals_preserve_all_inputs_and_borrow_scores() {
    let mut predictor = MlConformalPredictor::new(ConformalConfig {
        confidence_level: 0.9,
        calibration_size: 2,
    })
    .unwrap();
    let calibration_predictions = [
        Array2::from_elem((1, 1), 1.0_f32),
        Array2::from_elem((1, 1), 2.0_f32),
    ];
    let calibration_targets = [
        Array2::from_elem((1, 1), 1.1_f32),
        Array2::from_elem((1, 1), 2.2_f32),
    ];
    predictor
        .calibrate(&calibration_predictions, &calibration_targets)
        .unwrap();

    let predictions = [
        Array2::from_elem((1, 1), 3.0_f32),
        Array2::from_elem((1, 1), 5.0_f32),
    ];
    let result = predictor.predict_intervals(&predictions).unwrap();
    let interval = result
        .prediction_intervals
        .get("90%")
        .expect("90% is a configured interval family");
    assert_eq!(interval.lower.len(), 2);
    assert_eq!(interval.upper.len(), 2);
    assert!(
        (interval.lower[0][[0, 0]] - 2.8).abs() <= 4.0 * f32::EPSILON,
        "first lower endpoint must use the corrected maximum score"
    );
    assert!(
        (interval.upper[1][[0, 0]] - 5.2).abs() <= 8.0 * f32::EPSILON,
        "second upper endpoint must be retained"
    );
    assert!(
        matches!(result.conformity_scores, Cow::Borrowed(_)),
        "conformity scores must be borrowed"
    );
    assert_eq!(
        result.conformity_scores.as_ptr(),
        predictor.calibration_scores.as_ptr(),
        "borrowed scores must share the calibration buffer"
    );
}

#[test]
fn test_summary_and_zero_width_efficiency_preserve_undefined_state() {
    let mut predictor = MlConformalPredictor::new(ConformalConfig {
        confidence_level: 0.9,
        calibration_size: 1,
    })
    .unwrap();
    assert!(
        predictor.calibration_summary().score_distribution.is_none(),
        "an uncalibrated predictor has no score distribution"
    );

    let prediction = Array2::from_elem((1, 1), 2.0_f32);
    predictor
        .calibrate(
            std::slice::from_ref(&prediction),
            std::slice::from_ref(&prediction),
        )
        .unwrap();
    let metrics = predictor
        .validate_performance(
            std::slice::from_ref(&prediction),
            std::slice::from_ref(&prediction),
        )
        .unwrap();
    assert_eq!(metrics.mean_interval_width.to_bits(), 0.0_f32.to_bits());
    assert!(
        metrics.coverage_efficiency.is_none(),
        "coverage divided by zero width is undefined"
    );
}
