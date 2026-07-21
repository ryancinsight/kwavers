use super::*;
use crate::inverse::pinn::ml::{PinnConfig2D, PinnWave2D};

type TestBackend = coeus_core::MoiraiBackend;

fn model() -> PinnWave2D<TestBackend> {
    PinnWave2D::new(PinnConfig2D::default()).expect("valid PINN fixture")
}

#[test]
fn miscoverage_validation_preserves_provider_contract() {
    let error = PinnConformalPredictor::new(model(), 0.0).unwrap_err();
    assert!(
        format!("{error:?}").contains("miscoverage"),
        "zero miscoverage must report the provider boundary"
    );
}

#[test]
fn corrected_rank_selects_largest_of_five_native_scores() {
    let mut predictor = PinnConformalPredictor::new(model(), 0.1).unwrap();
    let input = vec![0.0_f32, 0.0, 0.0];
    let center = predictor.predict_scalar(&input).unwrap();
    let inputs = vec![input; 5];
    let targets = [0.1_f32, 0.2, 0.3, 0.4, 0.5].map(|offset| center + offset);

    predictor.calibrate(&inputs, &targets).unwrap();

    let largest = *predictor
        .calibration_scores()
        .last()
        .expect("successful calibration has scores");
    assert_eq!(
        predictor
            .calibrated_radius()
            .expect("successful calibration has a radius")
            .to_bits(),
        largest.to_bits(),
        "five scores at 90% coverage must select the finite-sample upper rank"
    );
}

#[test]
fn prediction_requires_calibration_and_finite_coordinates() {
    let mut predictor = PinnConformalPredictor::new(model(), 0.1).unwrap();
    let state_error = predictor.predict_conformal(&[0.0, 0.0, 0.0]).unwrap_err();
    assert!(
        format!("{state_error:?}").contains("calibrated before prediction"),
        "typestate error must state the missing calibration"
    );

    let finite_error = predictor
        .calibrate(&[vec![f32::NAN, 0.0, 0.0]], &[0.0])
        .unwrap_err();
    assert!(
        format!("{finite_error:?}").contains("coordinates must be finite"),
        "non-finite coordinate rejection must state its invariant"
    );
    assert_eq!(
        predictor.calibrated_radius(),
        None,
        "failed calibration must preserve the uncalibrated state"
    );
}
