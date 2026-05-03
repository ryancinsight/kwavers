use super::*;

#[test]
fn test_conformal_predictor_creation() {
    let config = ConformalConfig {
        confidence_level: 0.9,
        calibration_size: 100,
    };

    let predictor = ConformalPredictor::new(config);
    assert!(predictor.is_ok());
}

#[test]
fn test_conformal_calibration() {
    let config = ConformalConfig {
        confidence_level: 0.9,
        calibration_size: 10,
    };

    let mut predictor = ConformalPredictor::new(config).unwrap();

    let predictions = vec![
        Array2::from_elem((5, 5), 1.0),
        Array2::from_elem((5, 5), 2.0),
    ];
    let targets = vec![
        Array2::from_elem((5, 5), 1.1),
        Array2::from_elem((5, 5), 1.9),
    ];

    let result = predictor.calibrate(&predictions, &targets);
    assert!(result.is_ok());
    assert!(predictor.is_calibrated());
}

#[test]
fn test_conformity_score_computation() {
    let config = ConformalConfig::default();
    let predictor = ConformalPredictor::new(config).unwrap();

    let prediction = Array2::from_elem((3, 3), 1.0);
    let target = Array2::from_elem((3, 3), 1.2);

    let score = predictor.compute_conformity_score(&prediction, &target);
    assert!(score > 0.0);
}

#[test]
fn test_quantile_computation() {
    let config = ConformalConfig::default();
    let mut predictor = ConformalPredictor::new(config).unwrap();

    predictor.calibration_scores = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    predictor.is_calibrated = true;

    let quantile = predictor.compute_quantile(0.9);
    assert_eq!(quantile, 0.1);
}
