use kwavers_core::constants::fundamental::SOUND_SPEED_AIR;
use kwavers_core::constants::numerical::MHZ_TO_HZ;

use super::*;
use kwavers_core::constants::numerical::TWO_PI;
use ndarray::{Array2, Array3};

#[test]
fn test_physics_loss_config_default() {
    // default: sound_speed=343.0, frequency=1e6, lambda_data_init=0.8, lambda_physics_init=0.2
    let config = PhysicsLossConfig::default();
    config.validate().unwrap();
    assert!(
        config.sound_speed > 0.0,
        "default sound_speed must be positive, got {}",
        config.sound_speed
    );
    assert!(
        config.frequency > 0.0,
        "default frequency must be positive, got {}",
        config.frequency
    );
    assert!(
        config.history_window > 0,
        "default history_window must be > 0, got {}",
        config.history_window
    );
}

#[test]
fn test_physics_loss_config_validation() {
    let mut config = PhysicsLossConfig::default();
    config.sound_speed = 0.0;
    let err = config.validate().unwrap_err();
    assert!(
        format!("{err:?}").contains("sound_speed"),
        "zero sound_speed error must mention 'sound_speed'; got: {err:?}"
    );

    config.sound_speed = SOUND_SPEED_AIR;
    config.frequency = -100.0;
    let err = config.validate().unwrap_err();
    assert!(
        format!("{err:?}").contains("frequency"),
        "negative frequency error must mention 'frequency'; got: {err:?}"
    );

    config.frequency = MHZ_TO_HZ;
    config.history_window = 0;
    let err = config.validate().unwrap_err();
    assert!(
        format!("{err:?}").contains("history_window"),
        "zero history_window error must mention 'history_window'; got: {err:?}"
    );
}

#[test]
fn test_physics_loss_creation() {
    // k = 2πf/c = 2π·1e6/343 ≈ 18313 rad/m
    let config = PhysicsLossConfig::default();
    let loss = PhysicsInformedLoss::new(config).unwrap();
    let k_expected = TWO_PI * MHZ_TO_HZ / SOUND_SPEED_AIR;
    assert!(
        (loss.wave_number() - k_expected).abs() < 1.0,
        "wave_number = {} (expected ≈ {k_expected})",
        loss.wave_number()
    );
    assert!(loss.wave_number() > 0.0, "wave_number must be positive");
}

#[test]
fn test_wave_equation_residual_2d() {
    let config = PhysicsLossConfig::default();
    let loss = PhysicsInformedLoss::new(config).unwrap();

    let field = Array2::<f64>::zeros((5, 5));
    let residual = loss.wave_equation_residual_2d(&field);
    assert!((residual - 0.0).abs() < 1e-10);
}

#[test]
fn test_wave_equation_residual_3d() {
    let config = PhysicsLossConfig::default();
    let loss = PhysicsInformedLoss::new(config).unwrap();

    let field = Array3::<f64>::zeros((5, 5, 5));
    let residual = loss.wave_equation_residual_3d(&field);
    assert!((residual - 0.0).abs() < 1e-10);
}

#[test]
fn test_wave_number_computation() {
    let config = PhysicsLossConfig::default().with_wave_params(SOUND_SPEED_AIR, MHZ_TO_HZ);
    let loss = PhysicsInformedLoss::new(config).unwrap();

    // k = 2πf/c = 2π·1e6/343 ≈ 18313
    let k_expected = TWO_PI * MHZ_TO_HZ / SOUND_SPEED_AIR;
    assert!((loss.wave_number() - k_expected).abs() < 1.0);
}

#[test]
fn test_reciprocity_loss() {
    let forward =
        Array2::<f64>::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let reverse = forward.clone();

    let loss = PhysicsInformedLoss::reciprocity_loss(&forward, &reverse);
    assert!((loss - 0.0).abs() < 1e-10);
}

#[test]
fn test_reciprocity_loss_violation() {
    let forward = Array2::<f64>::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let reverse = Array2::<f64>::from_shape_vec((2, 2), vec![1.0, 2.0, 4.0, 5.0]).unwrap();

    let loss = PhysicsInformedLoss::reciprocity_loss(&forward, &reverse);
    assert!(loss > 0.0);
}

#[test]
fn test_coherence_loss_uniform_field() {
    let amplitudes = Array2::<f64>::from_elem((5, 5), 1.0);
    let phases = Array2::<f64>::from_elem((5, 5), 0.0);

    let loss = PhysicsInformedLoss::coherence_loss(&amplitudes, &phases);
    assert!((loss - 0.0).abs() < 1e-10);
}

#[test]
fn test_weight_schedule_exponential() {
    let config =
        PhysicsLossConfig::default().with_schedule(WeightSchedule::Exponential { decay_rate: 0.1 });
    let mut loss = PhysicsInformedLoss::new(config).unwrap();

    let (_lambda_data1, lambda_physics1) = loss.compute_weight_schedule(1.0, 1.0).unwrap();

    loss.current_epoch = 10;
    let (_lambda_data2, lambda_physics2) = loss.compute_weight_schedule(1.0, 1.0).unwrap();

    assert!(lambda_physics2 < lambda_physics1);
}

#[test]
fn test_weight_schedule_linear() {
    let config =
        PhysicsLossConfig::default().with_schedule(WeightSchedule::Linear { total_epochs: 100 });
    let mut loss = PhysicsInformedLoss::new(config).unwrap();

    let (_lambda_data1, lambda_physics1) = loss.compute_weight_schedule(1.0, 1.0).unwrap();

    loss.current_epoch = 50;
    let (_lambda_data2, lambda_physics2) = loss.compute_weight_schedule(1.0, 1.0).unwrap();

    assert!(lambda_physics2 < lambda_physics1);
}

#[test]
fn test_weight_schedule_adaptive() {
    let config = PhysicsLossConfig::default().with_schedule(WeightSchedule::Adaptive);
    let loss = PhysicsInformedLoss::new(config).unwrap();

    let (_, lambda_physics) = loss.compute_weight_schedule(1.0, 100.0).unwrap();
    assert!(lambda_physics < 0.3);

    let (_, lambda_physics) = loss.compute_weight_schedule(100.0, 1.0).unwrap();
    assert!(lambda_physics > 0.2);
}

#[test]
fn test_total_loss_computation() {
    let config = PhysicsLossConfig::default();
    let mut loss = PhysicsInformedLoss::new(config).unwrap();

    let total = loss.compute_total_loss(1.0, 1.0).unwrap();
    assert!(total > 0.0);
    assert!(total.is_finite());
}

#[test]
fn test_loss_history_tracking() {
    let config = PhysicsLossConfig::default();
    let mut loss = PhysicsInformedLoss::new(config).unwrap();

    loss.compute_total_loss(1.0, 1.0).unwrap();
    loss.compute_total_loss(0.9, 0.8).unwrap();
    loss.compute_total_loss(0.8, 0.7).unwrap();

    let history = loss.loss_history();
    assert_eq!(history.len(), 3);
    assert_eq!(history[0].epoch, 0);
    assert_eq!(history[1].epoch, 1);
    assert_eq!(history[2].epoch, 2);
}

#[test]
fn test_loss_history_window() {
    let config = PhysicsLossConfig::default();
    let mut loss = PhysicsInformedLoss::new(config).unwrap();

    for _ in 0..30 {
        loss.compute_total_loss(1.0, 1.0).unwrap();
    }

    let history = loss.loss_history();
    assert!(history.len() <= 20);
}

#[test]
fn test_reset() {
    let config = PhysicsLossConfig::default();
    let mut loss = PhysicsInformedLoss::new(config).unwrap();

    loss.compute_total_loss(1.0, 1.0).unwrap();
    loss.compute_total_loss(1.0, 1.0).unwrap();
    assert_eq!(loss.current_epoch(), 2);

    loss.reset();
    assert_eq!(loss.current_epoch(), 0);
    assert_eq!(loss.loss_history().len(), 0);
}

#[test]
fn test_builder_pattern() {
    let config = PhysicsLossConfig::default()
        .with_loss_weights(0.7, 0.3)
        .with_wave_params(400.0, 2.0 * MHZ_TO_HZ)
        .with_schedule(WeightSchedule::Linear { total_epochs: 50 })
        .without_history();

    assert!((config.lambda_data_init - 0.7).abs() < 1e-10);
    assert!((config.lambda_physics_init - 0.3).abs() < 1e-10);
    assert!((config.sound_speed - 400.0).abs() < 1e-10);
    assert!((config.frequency - 2.0 * MHZ_TO_HZ).abs() < 1e-10);
    assert!(!config.track_history);
}

// ─── Exact value-semantic tests ───────────────────────────────────────────────

/// `wave_equation_residual_2d` on a constant (all-1.0) field equals k⁴.
///
/// For constant u=1 at each interior point:
///   laplacian = (1+1+1+1) − 4·1 = 0
///   residual  = k²·1 + 0 = k²
///   residual² = k⁴
///   interior count = (5-2)² = 9 → mean = 9·k⁴/9 = k⁴
#[test]
fn wave_residual_2d_constant_field_equals_k_fourth() {
    let config = PhysicsLossConfig::default().with_wave_params(SOUND_SPEED_AIR, MHZ_TO_HZ);
    let loss_fn = PhysicsInformedLoss::new(config).unwrap();
    let k = loss_fn.wave_number();
    let expected = k.powi(4);
    let field = Array2::<f64>::ones((5, 5));
    let residual = loss_fn.wave_equation_residual_2d(&field);
    assert!(
        (residual - expected).abs() / expected < 1e-10,
        "constant-field 2D residual = {residual:.6e}, expected k⁴ = {expected:.6e}"
    );
}

/// `wave_equation_residual_3d` on a constant (all-1.0) field equals k⁴.
///
/// For constant u=1 at each interior point:
///   laplacian = (1+1+1+1+1+1) − 6·1 = 0
///   residual  = k²·1 + 0 = k²
///   residual² = k⁴
///   interior count = (5-2)³ = 27 → mean = 27·k⁴/27 = k⁴
#[test]
fn wave_residual_3d_constant_field_equals_k_fourth() {
    let config = PhysicsLossConfig::default().with_wave_params(SOUND_SPEED_AIR, MHZ_TO_HZ);
    let loss_fn = PhysicsInformedLoss::new(config).unwrap();
    let k = loss_fn.wave_number();
    let expected = k.powi(4);
    let field = Array3::<f64>::ones((5, 5, 5));
    let residual = loss_fn.wave_equation_residual_3d(&field);
    assert!(
        (residual - expected).abs() / expected < 1e-10,
        "constant-field 3D residual = {residual:.6e}, expected k⁴ = {expected:.6e}"
    );
}

/// `wave_equation_residual_2d` returns 0 for a 3×3 field (no interior points).
///
/// A 3×3 grid has interior range 1..2 which is empty; count=0 → early return 0.0.
#[test]
fn wave_residual_2d_too_small_returns_zero() {
    let config = PhysicsLossConfig::default();
    let loss_fn = PhysicsInformedLoss::new(config).unwrap();
    // 3×3 grid has interior range i∈1..2 and j∈1..2 → count=1 interior point
    // Use 2×2 grid for which i∈1..1 is empty → count=0
    let field = Array2::<f64>::ones((2, 2));
    let residual = loss_fn.wave_equation_residual_2d(&field);
    assert!(
        residual.abs() < 1e-14,
        "2×2 field has no interior points; residual must be 0.0, got {residual}"
    );
}

/// `reciprocity_loss` with exact known difference is MSE = ||Δ||² / N.
///
/// forward = [[1, 2], [3, 4]], reverse = zeros(2×2).
/// diff = [[1,2],[3,4]] → sum_sq = 1+4+9+16 = 30; N = 4 → result = 7.5.
#[test]
fn reciprocity_loss_exact_mse_nonzero() {
    let forward = Array2::<f64>::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let reverse = Array2::<f64>::zeros((2, 2));
    let loss = PhysicsInformedLoss::reciprocity_loss(&forward, &reverse);
    assert!(
        (loss - 7.5).abs() < 1e-14,
        "reciprocity_loss = {loss} (expected 7.5 = 30/4)"
    );
}

/// `reciprocity_loss` with mismatched dims returns infinity.
///
/// forward: (2,2), reverse: (2,3) → dims differ → f64::INFINITY.
#[test]
fn reciprocity_loss_mismatched_dims_is_infinity() {
    let forward = Array2::<f64>::zeros((2, 2));
    let reverse = Array2::<f64>::zeros((2, 3));
    let loss = PhysicsInformedLoss::reciprocity_loss(&forward, &reverse);
    assert!(
        loss.is_infinite(),
        "mismatched-dim reciprocity_loss must be ∞, got {loss}"
    );
}

/// `coherence_loss` with a uniform x-direction jump of 1.0 radian on 2×2 grid.
///
/// phases = [[0, 0], [1, 1]], amplitudes = ones(2×2).
/// First loop (x-adjacent pairs, 1×2 = 2 pairs): diff=1.0 (< π) → each sq = 1.0 → sum = 2.0.
/// Second loop (y-adjacent pairs, 2×1 = 2 pairs):
///   (i=0,j=0): diff=|0-0|=0 → 0; (i=1,j=0): diff=|1-1|=0 → 0 → sum = 0.
/// Total = 2.0; N = 2·((2-1)·2 + 2·(2-1)) = 2·(2+2) = 8 → result = 2.0/8 = 0.25.
#[test]
fn coherence_loss_row_jump_exact() {
    let amplitudes = Array2::<f64>::ones((2, 2));
    let phases = Array2::<f64>::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
    let loss = PhysicsInformedLoss::coherence_loss(&amplitudes, &phases);
    assert!(
        (loss - 0.25).abs() < 1e-14,
        "coherence_loss = {loss} (expected 0.25)"
    );
}

/// `coherence_loss` returns infinity for amplitude/phase shape mismatch.
///
/// amplitudes: (2,2), phases: (3,2) → dims differ → f64::INFINITY.
#[test]
fn coherence_loss_mismatched_dims_is_infinity() {
    let amplitudes = Array2::<f64>::ones((2, 2));
    let phases = Array2::<f64>::zeros((3, 2));
    let loss = PhysicsInformedLoss::coherence_loss(&amplitudes, &phases);
    assert!(
        loss.is_infinite(),
        "mismatched-dim coherence_loss must be ∞, got {loss}"
    );
}

/// `wave_number` formula: k = 2πf/c.  Exact IEEE 754 check.
///
/// c=343.0, f=MHZ_TO_HZ: k = 2π·1e6/343.
/// Computed in test independently; both must be bitwise equal after the
/// same floating-point operations.
#[test]
fn wave_number_exact_formula_verification() {
    let config = PhysicsLossConfig::default().with_wave_params(SOUND_SPEED_AIR, MHZ_TO_HZ);
    let loss_fn = PhysicsInformedLoss::new(config).unwrap();
    let expected_k = TWO_PI * MHZ_TO_HZ / SOUND_SPEED_AIR;
    assert!(
        (loss_fn.wave_number() - expected_k).abs() < 1e-10,
        "wave_number = {} (expected {expected_k})",
        loss_fn.wave_number()
    );
}
