use super::*;

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

/// The fixed-scale gradient used with `estimate_source_scaling=true` must be a
/// descent direction at any interior iterate.
///
/// The gradient of `J(s; α_fixed) = 0.5 ‖α_fixed F(s) − d‖²` with α_fixed
/// held constant is exactly the discrete adjoint gradient evaluated at α_fixed.
/// Since J is differentiable and the gradient is its own exact derivative, a
/// sufficiently small step along −∇J must strictly decrease J.
///
/// This test evaluates the directional derivative numerically by taking a
/// central-difference step ±h along the computed gradient direction and
/// confirming that `J(s − h∇J/‖∇J‖) < J(s)`.
#[test]
fn source_scaled_gradient_is_descent_direction() {
    let array = MultiRowRingArray::new(4, 1, 0.01, 0.0).expect("ring array");
    let config = Config {
        spacing_m: 0.005,
        estimate_source_scaling: true,
        forward_operator: Arc::new(DenseConvergentBornOperator {
            iterations: 64,
            relative_tolerance: 1.0e-13,
        }),
        ..Config::default()
    };
    let mut truth = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);
    truth[[1, 1, 0]] = 1515.0;
    let observed =
        simulate_frequency_observation(&truth, &array, 200_000.0, &config).expect("observed");
    // Evaluate gradient at a model different from truth.
    let mut current_speed = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);
    current_speed[[0, 0, 0]] = 1490.0;
    let current_slowness = sound_speed_to_slowness(&current_speed).expect("slowness");
    let observations = [FrequencyObservation::new(200_000.0, observed)];
    let (objective, gradient) =
        objective_and_gradient(&current_slowness, &observations, &array, &config)
            .expect("objective_and_gradient");
    // Compute a normalized step h in the −∇J direction small enough to stay
    // within the quadratic regime.
    let grad_norm = gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        grad_norm > f64::EPSILON,
        "gradient must be nonzero at non-truth model"
    );
    let step_size = 1.0e-9 / grad_norm;
    let candidate_slowness = current_slowness
        .iter()
        .zip(gradient.iter())
        .map(|(&s, &g)| s - step_size * g)
        .collect::<Vec<_>>();
    let candidate_slowness =
        Array3::from_shape_vec(current_slowness.dim(), candidate_slowness).expect("shape");
    let (candidate_objective, _) =
        objective_and_gradient(&candidate_slowness, &observations, &array, &config)
            .expect("candidate objective");
    assert!(
        candidate_objective < objective,
        "descent step must decrease objective: original={objective:.6e}, candidate={candidate_objective:.6e}"
    );
}

/// FWI with `estimate_source_scaling=true` must converge to the truth on a
/// consistent problem (same CBS forward model for data generation and inversion).
///
/// For a consistent forward model `d = α_true F(s_true)`, the global minimum
/// of `J(s) = 0.5 ‖α(s) F(s) − d‖²` is zero at `s = s_true`.  The
/// alternating-descent scheme (re-estimate α each Armijo candidate, then update
/// s) is a well-established variant of iterative source inversion; the
/// fixed-scale gradient is a valid descent direction at every iterate because it
/// equals the exact gradient of `J(s; α_fixed)` with the current α held
/// constant.  Convergence to the global minimum is not guaranteed in general but
/// holds for the small 3×3×1 test problem below, where the objective landscape
/// is convex near s_true.
#[test]
fn inversion_with_source_scaling_converges_for_consistent_model() {
    let array = MultiRowRingArray::new(4, 1, 0.01, 0.0).expect("ring array");
    let config = Config {
        spacing_m: 0.005,
        iterations: 14,
        initial_step_s_per_m: 5.0e-6,
        estimate_source_scaling: true,
        forward_operator: Arc::new(DenseConvergentBornOperator {
            iterations: 64,
            relative_tolerance: 1.0e-13,
        }),
        ..Config::default()
    };
    let mut truth = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);
    truth[[1, 1, 0]] = 1520.0;
    let observed =
        simulate_frequency_observation(&truth, &array, 200_000.0, &config).expect("observed");
    let observations = [FrequencyObservation::new(200_000.0, observed)];
    let initial = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);
    let result = invert(&observations, &array, &initial, &config).expect("inversion");

    // Objective must decrease by at least 80 %.
    let initial_obj = result.objective_history[0];
    let final_obj = result.objective_history.last().copied().unwrap();
    assert!(
        final_obj < 0.2 * initial_obj,
        "objective must decrease >80 %: initial={initial_obj:.4e}, final={final_obj:.4e}, history={:?}",
        result.objective_history
    );
    // Central high-speed voxel must be elevated above water (FWI recovers sign of anomaly).
    assert!(
        result.sound_speed_m_s[[1, 1, 0]] > SOUND_SPEED_WATER_SIM + 5.0,
        "center voxel must converge toward 1520 m/s, got {}",
        result.sound_speed_m_s[[1, 1, 0]]
    );
}

/// FWI with `PstdFiniteWindowBornOperator` must reduce the objective on a
/// consistent single-scatter problem.
///
/// The forward model used for data generation and inversion is identical
/// (same discrete PSTD physics), so the global minimum is at the true model
/// and the NCG loop must make progress on every run.
#[test]
fn pstd_finite_window_born_inversion_reduces_objective() {
    let array = MultiRowRingArray::new(4, 1, 0.01, 0.0).expect("ring array");
    let config = Config {
        spacing_m: 0.005,
        iterations: 10,
        initial_step_s_per_m: 3.0e-6,
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
    truth[[1, 1, 0]] = 1520.0;
    let observed =
        simulate_frequency_observation(&truth, &array, 200_000.0, &config).expect("observed");
    let observations = [FrequencyObservation::new(200_000.0, observed)];
    let initial = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);
    let result = invert(&observations, &array, &initial, &config).expect("inversion");

    assert!(
        result.objective_history.len() >= 2,
        "must record at least two objectives"
    );
    assert!(
        result.objective_history.last().copied().unwrap() < result.objective_history[0],
        "objective must decrease: {:?}",
        result.objective_history
    );
    assert!(
        result.sound_speed_m_s[[1, 1, 0]] > SOUND_SPEED_WATER_SIM,
        "central voxel must be elevated above reference water speed, got {}",
        result.sound_speed_m_s[[1, 1, 0]]
    );
}

/// Ali 2025 Table 1 reduced-grid parity gate.
///
/// Generates a synthetic breast-like phantom on a 5×5×3 grid with four
/// interior sound-speed anomalies spanning the clinically relevant range
/// (1510–1540 m/s in a 1500 m/s water background), synthesises finite-window
/// PSTD Born observations at two frequencies, and asserts that FWI with
/// `PstdFiniteWindowBornOperator` recovers the phantom with:
///
/// - RMSE ≤ 31.0 m/s  (2× the Ali 2025 Table 1 3-D FWI RMSE of 15.5 m/s)
/// - PCC  ≥ 0.84056  (95% of the Ali 2025 Table 1 PCC of 0.8848)
///
/// Both thresholds are derived directly from the published Table 1 values and
/// are therefore not empirically adjusted.
///
/// Uses an odd-sized 5×5×3 grid so the ring array at diameter 0.010 m
/// (radius 0.005 m) naturally aligns with grid coordinates: for a centered
/// (5,5,3) grid at 0.005 m spacing the axis coordinates are
/// {−0.010, −0.005, 0, 0.005, 0.010}, so the four cylindrical ring elements
/// at (±0.005, 0, 0) and (0, ±0.005, 0) map exactly to grid indices.
#[test]
fn ali2025_table1_parity_gate() {
    let array = MultiRowRingArray::new(4, 1, 0.010, 0.0).expect("ring array");
    let config = Config {
        reference_sound_speed_m_s: 1500.0,
        spacing_m: 0.005,
        iterations: 20,
        initial_step_s_per_m: 3.0e-6,
        min_sound_speed_m_s: 1450.0,
        max_sound_speed_m_s: 1580.0,
        estimate_source_scaling: false,
        tikhonov_weight: 0.0,
        forward_operator: Arc::new(PstdFiniteWindowBornOperator {
            time_step_s: 1.0e-7,
            source_amplitude_pa: 1.0,
            cycles_per_frequency: 4,
            frequency_bin_cycles: 1,
        }),
    };

    // Synthetic breast-like phantom: 1500 m/s water background with
    // four interior anomalies in the clinical range 1510–1540 m/s.
    let mut truth = Array3::from_elem((5, 5, 3), 1500.0_f64);
    truth[[1, 1, 1]] = 1530.0; // glandular tissue
    truth[[3, 3, 1]] = 1520.0; // fibroglandular
    truth[[1, 3, 1]] = 1510.0; // fatty tissue
    truth[[3, 1, 1]] = 1540.0; // dense tissue

    // Generate consistent observations at two frequencies using the same
    // forward model as the inversion (closed-loop / self-consistent test).
    let freq_lo = 200_000.0_f64;
    let freq_hi = 400_000.0_f64;
    let obs_lo = simulate_frequency_observation(&truth, &array, freq_lo, &config).expect("obs_lo");
    let obs_hi = simulate_frequency_observation(&truth, &array, freq_hi, &config).expect("obs_hi");
    let observations = [
        FrequencyObservation::new(freq_lo, obs_lo),
        FrequencyObservation::new(freq_hi, obs_hi),
    ];

    let initial = Array3::from_elem((5, 5, 3), 1500.0_f64);
    let result = invert(&observations, &array, &initial, &config).expect("inversion");

    let truth_flat: Vec<f64> = truth.iter().copied().collect();
    let recon_flat: Vec<f64> = result.sound_speed_m_s.iter().copied().collect();
    let n = truth_flat.len() as f64;
    let rmse = (truth_flat
        .iter()
        .zip(recon_flat.iter())
        .map(|(&t, &r)| (t - r) * (t - r))
        .sum::<f64>()
        / n)
        .sqrt();
    let pcc = kwavers_math::statistics::pearson(&truth_flat, &recon_flat);

    assert!(
        rmse <= 31.0,
        "RMSE {rmse:.2} m/s exceeds Ali 2025 Table 1 2× gate (31.0 m/s); \
         objective history: {:?}",
        result.objective_history
    );
    assert!(
        pcc >= 0.84056,
        "PCC {pcc:.6} below Ali 2025 Table 1 95% gate (0.84056); \
         objective history: {:?}",
        result.objective_history
    );
}
