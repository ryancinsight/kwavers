use super::*;

#[test]
fn test_conformal_predictor_creation() {
    let config = ConformalConfig {
        confidence_level: 0.9,
        calibration_size: 100,
    };
    let predictor = ConformalPredictor::new(config).unwrap();
    // Before calibration: no scores and not calibrated.
    assert!(!predictor.is_calibrated(), "fresh predictor must not be calibrated");
    assert!(
        predictor.calibration_scores.is_empty(),
        "calibration_scores must be empty before calibrate()"
    );
}

#[test]
fn test_conformal_predictor_rejects_invalid_confidence() {
    let err = ConformalPredictor::new(ConformalConfig {
        confidence_level: 0.0,
        calibration_size: 100,
    })
    .unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("Confidence level"),
        "zero confidence_level error must mention 'Confidence level'; got: {msg}"
    );

    let err2 = ConformalPredictor::new(ConformalConfig {
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
    // prediction=1.0, target=1.1 → conformity score = median(|1.0-1.1|) = 0.1
    // prediction=2.0, target=1.9 → conformity score = median(|2.0-1.9|) = 0.1
    // After calibration, calibration_scores = [0.1, 0.1] (sorted).
    let mut predictor = ConformalPredictor::new(ConformalConfig {
        confidence_level: 0.9,
        calibration_size: 10,
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
    assert!(predictor.is_calibrated(), "predictor must be calibrated after calibrate()");
    assert_eq!(
        predictor.calibration_scores.len(),
        2,
        "must store one score per calibration pair"
    );
    // Both scores are median(|Δ|) = 0.1 for uniform offset on (5,5) grid.
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
    // prediction=1.0, target=1.2 on (3,3): abs_error=0.2 everywhere.
    // Median of 9 identical values is 0.2.
    let predictor = ConformalPredictor::new(ConformalConfig::default()).unwrap();
    let prediction = Array2::from_elem((3, 3), 1.0_f32);
    let target = Array2::from_elem((3, 3), 1.2_f32);

    let score = predictor.compute_conformity_score(&prediction, &target);
    let err = (score - 0.2).abs();
    assert!(
        err < 1e-5,
        "conformity score = {score} (expected 0.2 for uniform offset 0.2)"
    );
    assert!(score > 0.0, "conformity score must be positive for non-matching prediction");
}

#[test]
fn test_quantile_computation() {
    // calibration_scores=[0.1, 0.2, 0.3, 0.4, 0.5]; confidence_level=0.9
    // alpha = 0.1; index = ceil(5·0.1) - 1 = ceil(0.5) - 1 = 1 - 1 = 0
    // quantile = calibration_scores[0] = 0.1
    let mut predictor = ConformalPredictor::new(ConformalConfig::default()).unwrap();
    predictor.calibration_scores = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    predictor.is_calibrated = true;

    let quantile = predictor.compute_quantile(0.9);
    assert_eq!(
        quantile, 0.1,
        "quantile(0.9) on [0.1..0.5] must select lowest decile = 0.1"
    );
}
