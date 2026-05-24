use super::*;

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
            temporal_transfer: None,
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
