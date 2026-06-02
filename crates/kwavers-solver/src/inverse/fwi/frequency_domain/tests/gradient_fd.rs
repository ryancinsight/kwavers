use super::*;

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
            temporal_transfer: None,
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

/// Discrete adjoint gradient for the finite-window PSTD Born operator must
/// match the centred finite-difference approximation to within 5×10⁻⁴
/// relative tolerance.
///
/// The adjoint is the exact gradient of the discrete PSTD Born objective
/// `J = 0.5 ‖F(s) − d‖²`, so the centred difference
/// `(J(s + ε e_j) − J(s − ε e_j)) / (2ε)` must agree with `∂J/∂sⱼ`
/// to within floating-point rounding and temporal discretisation error.
#[test]
fn pstd_finite_window_born_adjoint_gradient_matches_finite_difference() {
    let array = MultiRowRingArray::new(4, 1, 0.01, 0.0).expect("ring array");
    let config = Config {
        spacing_m: 0.005,
        estimate_source_scaling: false,
        forward_operator: Arc::new(PstdFiniteWindowBornOperator {
            time_step_s: 1.0e-7,
            source_amplitude_pa: 1.0,
            cycles_per_frequency: 4,
            frequency_bin_cycles: 1,
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
        .expect("finite-window objective gradient");

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
        "finite_difference={finite_difference:.6e}, analytic={:.6e}",
        gradient[[1, 1, 0]]
    );
}
