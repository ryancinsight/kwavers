//! Matrix-free LSQR damped reconstruction tests.
//!
//! These tests exercise the [`ShiftPrior::Lsqr`] path end-to-end through the
//! public [`reconstruct_sound_speed_shift`] / [`reconstruct_sound_speed_shift_with_workspace`]
//! APIs.

use ndarray::Array2;

use super::{
    attach_time_shifts, horizontal_samples, predict_sound_speed_time_shifts,
    reconstruct_sound_speed_shift, reconstruct_sound_speed_shift_with_workspace, ShiftPrior,
    SoundSpeedShiftConfig, SoundSpeedShiftWorkspace,
};

/// Straight-ray consistent system: 3×3 all-active grid, 3 horizontal rays,
/// uniform +20 m/s shift.  LSQR minimum-norm solution must recover 20 m/s in
/// every cell (±1e-3 m/s tolerance).
///
/// For each horizontal block: A = [L, L, L] (L = spacing_m).  The minimum
/// 2-norm solution to L·Σxᵢ = −c₀²·Δt is xᵢ = Δc for all i (uniform
/// allocation), which for the forward-consistent Δt gives xᵢ = 20 m/s.
#[test]
fn lsqr_prior_recovers_uniform_sound_speed_shift() {
    let mask = Array2::from_elem((3, 3), true);
    let mut truth = Array2::zeros((3, 3));
    truth.fill(20.0);
    let samples = horizontal_samples(&[-0.001, 0.0, 0.001]);
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        iterations: 64,
        prior: ShiftPrior::Lsqr { damping: 0.0 },
        tikhonov_weight: 0.0,
        smoothness_weight: 0.0,
        ..Default::default()
    };
    let predicted = predict_sound_speed_time_shifts(&truth, &samples, &mask, config).unwrap();
    let measured = attach_time_shifts(&samples, &predicted);

    let image = reconstruct_sound_speed_shift(&measured, &mask, config).unwrap();

    assert_eq!(image.active_voxels, 9);
    assert_eq!(image.rows_used, 3);
    for value in &image.sound_speed_shift_m_s {
        assert!(
            (*value - 20.0).abs() <= 1.0e-3,
            "LSQR reconstruction value {value:.6} differs from 20 m/s by more than 1e-3"
        );
    }
}

/// Increasing Tikhonov damping λ shrinks solution norm: ‖x_high‖₂ < ‖x_low‖₂.
///
/// For the augmented system `min ‖Ax−b‖²+λ²‖x‖²`, the solution satisfies
/// ‖x(λ)‖ = ‖(AᵀA+λ²I)⁻¹Aᵀb‖, which is strictly decreasing in λ for
/// non-zero b (SVD representation: ‖x(λ)‖² = Σ σᵢ²φᵢ²/(σᵢ²+λ²)²,
/// each term strictly decreasing in λ).
#[test]
fn lsqr_higher_damping_reduces_solution_norm() {
    let mask = Array2::from_elem((3, 3), true);
    let mut truth = Array2::zeros((3, 3));
    truth.fill(20.0);
    let samples = horizontal_samples(&[-0.001, 0.0, 0.001]);
    let base_config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        iterations: 128,
        tikhonov_weight: 0.0,
        smoothness_weight: 0.0,
        ..Default::default()
    };
    let predicted = predict_sound_speed_time_shifts(&truth, &samples, &mask, base_config).unwrap();
    let measured = attach_time_shifts(&samples, &predicted);

    let low_damping_image = reconstruct_sound_speed_shift(
        &measured,
        &mask,
        SoundSpeedShiftConfig {
            prior: ShiftPrior::Lsqr { damping: 1.0e-4 },
            ..base_config
        },
    )
    .unwrap();
    let high_damping_image = reconstruct_sound_speed_shift(
        &measured,
        &mask,
        SoundSpeedShiftConfig {
            prior: ShiftPrior::Lsqr { damping: 100.0 },
            ..base_config
        },
    )
    .unwrap();

    let norm_low = low_damping_image
        .sound_speed_shift_m_s
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    let norm_high = high_damping_image
        .sound_speed_shift_m_s
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();

    assert!(
        norm_low > 0.0,
        "low-damping solution must be nonzero, got {norm_low}"
    );
    assert!(
        norm_high < norm_low,
        "higher damping must reduce solution norm: ‖x_high‖={norm_high:.6}, ‖x_low‖={norm_low:.6}"
    );
}

/// `objective_history` (0.5·φ̄²) is non-increasing across LSQR iterations.
///
/// `φ̄` is updated as `φ̄ ← s·φ̄` where `s = β_{k+1}/ρ_k < 1` (since
/// `ρ_k = √(ρ̄_k²+β_{k+1}²+λ²) > β_{k+1}`), so the running residual estimate
/// decreases monotonically after initialisation.
#[test]
fn lsqr_objective_history_is_non_increasing() {
    let mask = Array2::from_elem((3, 3), true);
    let mut truth = Array2::zeros((3, 3));
    truth.fill(20.0);
    let samples = horizontal_samples(&[-0.001, 0.0, 0.001]);
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        iterations: 16,
        prior: ShiftPrior::Lsqr { damping: 0.0 },
        tikhonov_weight: 0.0,
        smoothness_weight: 0.0,
        ..Default::default()
    };
    let predicted = predict_sound_speed_time_shifts(&truth, &samples, &mask, config).unwrap();
    let measured = attach_time_shifts(&samples, &predicted);

    let image = reconstruct_sound_speed_shift(&measured, &mask, config).unwrap();

    let history = &image.objective_history;
    assert!(
        history.len() >= 2,
        "objective_history must have at least 2 entries, got {}",
        history.len()
    );
    for window in history.windows(2) {
        let (prev, next) = (window[0], window[1]);
        assert!(
            next <= prev + f64::EPSILON,
            "objective not non-increasing at transition {prev:.6e} → {next:.6e}"
        );
    }
}

/// Workspace slot count and byte footprint are preserved across two successive
/// LSQR calls with identical geometry.  Solution vectors must be bitwise equal.
#[test]
fn lsqr_workspace_capacity_preserved_across_calls() {
    let mask = Array2::from_elem((3, 3), true);
    let mut truth = Array2::zeros((3, 3));
    truth.fill(20.0);
    let samples = horizontal_samples(&[-0.001, 0.0, 0.001]);
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        iterations: 32,
        prior: ShiftPrior::Lsqr { damping: 1.0e-4 },
        tikhonov_weight: 0.0,
        smoothness_weight: 0.0,
        ..Default::default()
    };
    let predicted = predict_sound_speed_time_shifts(&truth, &samples, &mask, config).unwrap();
    let measured = attach_time_shifts(&samples, &predicted);
    let mut workspace = SoundSpeedShiftWorkspace::new();

    let first =
        reconstruct_sound_speed_shift_with_workspace(&measured, &mask, config, &mut workspace)
            .unwrap();
    let slots_after_first = workspace.allocated_slots();
    let bytes_after_first = workspace.memory_bytes();

    assert!(
        slots_after_first > 0,
        "workspace must allocate buffers on first LSQR call"
    );

    let second =
        reconstruct_sound_speed_shift_with_workspace(&measured, &mask, config, &mut workspace)
            .unwrap();

    assert_eq!(
        workspace.allocated_slots(),
        slots_after_first,
        "workspace slot count changed between calls"
    );
    assert_eq!(
        workspace.memory_bytes(),
        bytes_after_first,
        "workspace byte footprint changed between calls"
    );
    assert_eq!(
        first.sound_speed_shift_m_s, second.sound_speed_shift_m_s,
        "LSQR solutions differ between identical successive calls"
    );
}

/// Zero-data RHS (all time shifts = 0) triggers the early-exit path in LSQR
/// (`β₁ < 1e-12`) and returns a zero solution without panic.
///
/// `b = −c₀²·Δt = 0` for Δt = 0 ⟹ ‖b‖ < ε ⟹ LSQR returns x = 0
/// immediately after checking the `beta < 1e-12` guard.
#[test]
fn lsqr_zero_rhs_returns_zero_solution() {
    let mask = Array2::from_elem((3, 3), true);
    // All time_shift_s = 0 ⟹ b = −c₀²·0 = 0
    let samples = horizontal_samples(&[-0.001, 0.0, 0.001]);
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        iterations: 32,
        prior: ShiftPrior::Lsqr { damping: 0.0 },
        tikhonov_weight: 0.0,
        smoothness_weight: 0.0,
        ..Default::default()
    };

    let image = reconstruct_sound_speed_shift(&samples, &mask, config).unwrap();

    assert_eq!(
        image.active_voxels, 9,
        "active voxel count must be 9 even for zero RHS"
    );
    for value in &image.sound_speed_shift_m_s {
        assert!(
            value.abs() < 1.0e-12,
            "zero RHS must yield zero solution, got {value:.3e}"
        );
    }
}
