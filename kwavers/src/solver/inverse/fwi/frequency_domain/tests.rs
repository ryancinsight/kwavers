use std::sync::Arc;

use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;

use super::gradient::objective_and_gradient;
use super::{
    invert, simulate_frequency_observation, AbsorbingBoundary, Config, DenseConvergentBornOperator,
    FrequencyObservation, PstdSpectralConvergentBornOperator, SingleScatterBornOperator,
    SpectralConvergentBornOperator,
};
use crate::physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::{
    sound_speed_to_slowness, MultiRowRingArray,
};
use ndarray::Array3;

fn test_config() -> Config {
    Config {
        reference_sound_speed_m_s: SOUND_SPEED_WATER_SIM,
        spacing_m: 0.005,
        iterations: 6,
        initial_step_s_per_m: 3.0e-6,
        min_sound_speed_m_s: 1450.0,
        max_sound_speed_m_s: 1560.0,
        estimate_source_scaling: false,
        tikhonov_weight: 0.0,
        forward_operator: Arc::new(SingleScatterBornOperator),
    }
}

fn test_array() -> MultiRowRingArray {
    MultiRowRingArray::new(6, 2, 0.08, 0.01).expect("ring array")
}

#[test]
fn dense_cbs_prediction_matches_born_for_homogeneous_on_grid_ring() {
    let array = MultiRowRingArray::new(4, 1, 0.01, 0.0).expect("ring array");
    let model = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);
    let born = Config {
        spacing_m: 0.005,
        forward_operator: Arc::new(SingleScatterBornOperator),
        ..Config::default()
    };
    let cbs = Config {
        spacing_m: 0.005,
        forward_operator: Arc::new(DenseConvergentBornOperator {
            iterations: 6,
            relative_tolerance: 1.0e-12,
        }),
        ..Config::default()
    };

    let born_data = simulate_frequency_observation(&model, &array, 250_000.0, &born).expect("born");
    let cbs_data = simulate_frequency_observation(&model, &array, 250_000.0, &cbs).expect("cbs");
    let max_reference = born_data
        .iter()
        .map(|value| value.norm())
        .fold(0.0, f64::max);
    let max_error = born_data
        .iter()
        .zip(cbs_data.iter())
        .map(|(&lhs, &rhs)| (lhs - rhs).norm())
        .fold(0.0, f64::max);

    assert!(
        max_error <= f64::EPSILON.sqrt() * max_reference.max(1.0),
        "max_error={max_error}, max_reference={max_reference}"
    );
}

#[test]
fn dense_cbs_prediction_is_sensitive_to_sound_speed_volume() {
    let array = MultiRowRingArray::new(4, 1, 0.01, 0.0).expect("ring array");
    let base = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);
    let mut perturbed = base.clone();
    perturbed[[1, 1, 0]] = 1510.0;
    let config = Config {
        spacing_m: 0.005,
        forward_operator: Arc::new(DenseConvergentBornOperator {
            iterations: 8,
            relative_tolerance: 1.0e-12,
        }),
        ..Config::default()
    };

    let base_data =
        simulate_frequency_observation(&base, &array, 250_000.0, &config).expect("base");
    let perturbed_data =
        simulate_frequency_observation(&perturbed, &array, 250_000.0, &config).expect("perturbed");
    let max_reference = base_data
        .iter()
        .map(|value| value.norm())
        .fold(0.0, f64::max);
    let max_difference = base_data
        .iter()
        .zip(perturbed_data.iter())
        .map(|(&lhs, &rhs)| (lhs - rhs).norm())
        .fold(0.0, f64::max);

    assert!(
        max_difference > f64::EPSILON.sqrt() * max_reference.max(1.0),
        "max_difference={max_difference}, max_reference={max_reference}"
    );
}

#[test]
fn spectral_cbs_prediction_is_sensitive_to_sound_speed_volume() {
    let array = MultiRowRingArray::new(4, 1, 0.01, 0.0).expect("ring array");
    let base = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);
    let mut perturbed = base.clone();
    perturbed[[1, 1, 0]] = 1510.0;
    let config = Config {
        spacing_m: 0.005,
        forward_operator: Arc::new(SpectralConvergentBornOperator {
            iterations: 12,
            relative_tolerance: 1.0e-12,
            absorbing_boundary: AbsorbingBoundary::disabled(),
        }),
        ..Config::default()
    };

    let base_data =
        simulate_frequency_observation(&base, &array, 180_000.0, &config).expect("base");
    let perturbed_data =
        simulate_frequency_observation(&perturbed, &array, 180_000.0, &config).expect("perturbed");
    let max_difference = base_data
        .iter()
        .zip(perturbed_data.iter())
        .map(|(&lhs, &rhs)| (lhs - rhs).norm())
        .fold(0.0, f64::max);

    assert!(
        max_difference > 1.0e-9,
        "spectral CBS must respond to sound-speed changes"
    );
}

#[test]
fn pstd_spectral_cbs_prediction_is_sensitive_to_sound_speed_volume() {
    let array = MultiRowRingArray::new(4, 1, 0.01, 0.0).expect("ring array");
    let base = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);
    let mut perturbed = base.clone();
    perturbed[[1, 1, 0]] = 1510.0;
    let config = Config {
        spacing_m: 0.005,
        forward_operator: Arc::new(PstdSpectralConvergentBornOperator {
            iterations: 12,
            relative_tolerance: 1.0e-12,
            time_step_s: 1.0e-7,
            absorbing_boundary: AbsorbingBoundary::disabled(),
        }),
        ..Config::default()
    };

    let base_data =
        simulate_frequency_observation(&base, &array, 180_000.0, &config).expect("base");
    let perturbed_data =
        simulate_frequency_observation(&perturbed, &array, 180_000.0, &config).expect("perturbed");
    let max_difference = base_data
        .iter()
        .zip(perturbed_data.iter())
        .map(|(&lhs, &rhs)| (lhs - rhs).norm())
        .fold(0.0, f64::max);

    assert!(
        max_difference > 1.0e-9,
        "PSTD spectral CBS must respond to sound-speed changes"
    );
}

#[test]
fn dense_cbs_prediction_rejects_ring_outside_inversion_grid() {
    let array = MultiRowRingArray::new(4, 1, 0.10, 0.0).expect("ring array");
    let model = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);
    let config = Config {
        spacing_m: 0.005,
        forward_operator: Arc::new(DenseConvergentBornOperator {
            iterations: 2,
            relative_tolerance: 1.0e-8,
        }),
        ..Config::default()
    };

    let error = simulate_frequency_observation(&model, &array, 250_000.0, &config)
        .expect_err("outside ring must fail");
    assert!(
        error.to_string().contains("outside the inversion grid"),
        "{error}"
    );
}

#[test]
fn forward_model_is_sensitive_to_sound_speed_volume() {
    let array = test_array();
    let config = test_config();
    let base = Array3::from_elem((2, 2, 2), SOUND_SPEED_WATER_SIM);
    let mut perturbed = base.clone();
    perturbed[[1, 1, 1]] = 1530.0;

    let base_data =
        simulate_frequency_observation(&base, &array, 250_000.0, &config).expect("base data");
    let perturbed_data = simulate_frequency_observation(&perturbed, &array, 250_000.0, &config)
        .expect("perturbed data");
    let difference = base_data
        .iter()
        .zip(perturbed_data.iter())
        .map(|(&a, &b)| (a - b).norm())
        .fold(0.0, f64::max);

    assert!(
        difference > 1.0e-6,
        "sound-speed perturbation must alter frequency-domain pressure"
    );
}

#[test]
fn dense_cbs_adjoint_gradient_matches_finite_difference() {
    let array = MultiRowRingArray::new(4, 1, 0.01, 0.0).expect("ring array");
    let config = Config {
        spacing_m: 0.005,
        estimate_source_scaling: false,
        forward_operator: Arc::new(DenseConvergentBornOperator {
            iterations: 96,
            relative_tolerance: 1.0e-13,
        }),
        ..Config::default()
    };
    let mut truth = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);
    truth[[2, 1, 0]] = 1510.0;
    let observed =
        simulate_frequency_observation(&truth, &array, 180_000.0, &config).expect("observed");
    let observations = [FrequencyObservation::new(
        180_000.0,
        observed.slice(ndarray::s![0..2, ..]).to_owned(),
    )];
    let mut current_speed = Array3::from_elem((3, 3, 1), 1501.0);
    current_speed[[0, 0, 0]] = 1490.0;
    let current_slowness = sound_speed_to_slowness(&current_speed).expect("slowness");
    let (_, gradient) = objective_and_gradient(&current_slowness, &observations, &array, &config)
        .expect("dense objective gradient");

    let epsilon = 1.0e-8;
    let mut plus = current_slowness.clone();
    let mut minus = current_slowness.clone();
    plus[[1, 1, 0]] += epsilon;
    minus[[1, 1, 0]] -= epsilon;
    let (objective_plus, _) =
        objective_and_gradient(&plus, &observations, &array, &config).expect("plus");
    let (objective_minus, _) =
        objective_and_gradient(&minus, &observations, &array, &config).expect("minus");
    let finite_difference = (objective_plus - objective_minus) / (2.0 * epsilon);

    assert!(
        (finite_difference - gradient[[1, 1, 0]]).abs()
            <= 5.0e-4 * finite_difference.abs().max(1.0),
        "finite_difference={finite_difference}, analytic={}",
        gradient[[1, 1, 0]]
    );
}

#[test]
fn spectral_cbs_adjoint_gradient_matches_finite_difference() {
    let array = MultiRowRingArray::new(4, 1, 0.01, 0.0).expect("ring array");
    let config = Config {
        spacing_m: 0.005,
        estimate_source_scaling: false,
        forward_operator: Arc::new(SpectralConvergentBornOperator {
            iterations: 128,
            relative_tolerance: 1.0e-13,
            absorbing_boundary: AbsorbingBoundary::disabled(),
        }),
        ..Config::default()
    };
    let mut truth = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);
    truth[[2, 1, 0]] = 1510.0;
    let observed =
        simulate_frequency_observation(&truth, &array, 180_000.0, &config).expect("observed");
    let observations = [FrequencyObservation::new(
        180_000.0,
        observed.slice(ndarray::s![0..2, ..]).to_owned(),
    )];
    let mut current_speed = Array3::from_elem((3, 3, 1), 1501.0);
    current_speed[[0, 0, 0]] = 1490.0;
    let current_slowness = sound_speed_to_slowness(&current_speed).expect("slowness");
    let (_, gradient) = objective_and_gradient(&current_slowness, &observations, &array, &config)
        .expect("spectral objective gradient");

    let epsilon = 1.0e-8;
    let mut plus = current_slowness.clone();
    let mut minus = current_slowness.clone();
    plus[[1, 1, 0]] += epsilon;
    minus[[1, 1, 0]] -= epsilon;
    let (objective_plus, _) =
        objective_and_gradient(&plus, &observations, &array, &config).expect("plus");
    let (objective_minus, _) =
        objective_and_gradient(&minus, &observations, &array, &config).expect("minus");
    let finite_difference = (objective_plus - objective_minus) / (2.0 * epsilon);

    assert!(
        (finite_difference - gradient[[1, 1, 0]]).abs()
            <= 5.0e-4 * finite_difference.abs().max(1.0),
        "finite_difference={finite_difference}, analytic={}",
        gradient[[1, 1, 0]]
    );
}

#[test]
fn pstd_spectral_cbs_adjoint_gradient_matches_finite_difference() {
    let array = MultiRowRingArray::new(4, 1, 0.01, 0.0).expect("ring array");
    let config = Config {
        spacing_m: 0.005,
        estimate_source_scaling: false,
        forward_operator: Arc::new(PstdSpectralConvergentBornOperator {
            iterations: 128,
            relative_tolerance: 1.0e-13,
            time_step_s: 1.0e-7,
            absorbing_boundary: AbsorbingBoundary::disabled(),
        }),
        ..Config::default()
    };
    let mut truth = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);
    truth[[2, 1, 0]] = 1510.0;
    let observed =
        simulate_frequency_observation(&truth, &array, 180_000.0, &config).expect("observed");
    let observations = [FrequencyObservation::new(
        180_000.0,
        observed.slice(ndarray::s![0..2, ..]).to_owned(),
    )];
    let mut current_speed = Array3::from_elem((3, 3, 1), 1501.0);
    current_speed[[0, 0, 0]] = 1490.0;
    let current_slowness = sound_speed_to_slowness(&current_speed).expect("slowness");
    let (_, gradient) = objective_and_gradient(&current_slowness, &observations, &array, &config)
        .expect("PSTD spectral objective gradient");

    let epsilon = 1.0e-8;
    let mut plus = current_slowness.clone();
    let mut minus = current_slowness.clone();
    plus[[1, 1, 0]] += epsilon;
    minus[[1, 1, 0]] -= epsilon;
    let (objective_plus, _) =
        objective_and_gradient(&plus, &observations, &array, &config).expect("plus");
    let (objective_minus, _) =
        objective_and_gradient(&minus, &observations, &array, &config).expect("minus");
    let finite_difference = (objective_plus - objective_minus) / (2.0 * epsilon);

    assert!(
        (finite_difference - gradient[[1, 1, 0]]).abs()
            <= 5.0e-4 * finite_difference.abs().max(1.0),
        "finite_difference={finite_difference}, analytic={}",
        gradient[[1, 1, 0]]
    );
}

#[test]
fn adjoint_gradient_matches_finite_difference() {
    let array = test_array();
    let config = test_config();
    let mut truth = Array3::from_elem((2, 2, 2), SOUND_SPEED_WATER_SIM);
    truth[[1, 0, 1]] = 1520.0;
    let observed =
        simulate_frequency_observation(&truth, &array, 220_000.0, &config).expect("observed");
    let observations = [FrequencyObservation::new(
        220_000.0,
        observed.slice(ndarray::s![0..3, ..]).to_owned(),
    )];
    let current_speed = Array3::from_elem((2, 2, 2), 1502.0);
    let current_slowness = sound_speed_to_slowness(&current_speed).expect("slowness");
    let (objective, gradient) =
        objective_and_gradient(&current_slowness, &observations, &array, &config)
            .expect("objective gradient");

    let epsilon = 1.0e-8;
    let mut plus = current_slowness.clone();
    let mut minus = current_slowness.clone();
    plus[[1, 0, 1]] += epsilon;
    minus[[1, 0, 1]] -= epsilon;
    let (objective_plus, _) =
        objective_and_gradient(&plus, &observations, &array, &config).expect("plus");
    let (objective_minus, _) =
        objective_and_gradient(&minus, &observations, &array, &config).expect("minus");
    let finite_difference = (objective_plus - objective_minus) / (2.0 * epsilon);

    assert!(objective > 0.0);
    assert!(
        (finite_difference - gradient[[1, 0, 1]]).abs()
            <= 1.0e-5 * finite_difference.abs().max(1.0),
        "finite_difference={finite_difference}, analytic={}",
        gradient[[1, 0, 1]]
    );
}

#[test]
fn nonlinear_inversion_reduces_objective_and_raises_high_speed_target() {
    let array = test_array();
    let config = test_config();
    let mut truth = Array3::from_elem((3, 3, 3), SOUND_SPEED_WATER_SIM);
    truth[[1, 1, 1]] = 1535.0;
    let observed =
        simulate_frequency_observation(&truth, &array, 240_000.0, &config).expect("observed");
    let observations = [FrequencyObservation::new(
        240_000.0,
        observed.slice(ndarray::s![0..4, ..]).to_owned(),
    )];
    let initial = Array3::from_elem((3, 3, 3), SOUND_SPEED_WATER_SIM);

    let result = invert(&observations, &array, &initial, &config).expect("inversion");

    assert!(result.objective_history.len() >= 2);
    assert!(
        result.objective_history.last().copied().unwrap() < result.objective_history[0],
        "objective history={:?}",
        result.objective_history
    );
    assert!(
        result.sound_speed_m_s[[1, 1, 1]] > SOUND_SPEED_WATER_SIM,
        "central target speed={}",
        result.sound_speed_m_s[[1, 1, 1]]
    );
}
