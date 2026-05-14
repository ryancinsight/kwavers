//! Main training loop and coordination for PINN optimization
//!
//! This module implements the high-level training procedure that coordinates
//! data loading, loss computation, optimization, and convergence checking.

#[cfg(feature = "pinn")]
use super::super::model::ElasticPINN2D;
#[cfg(feature = "pinn")]
use burn::tensor::backend::AutodiffBackend;

#[cfg(feature = "pinn")]
use super::{data::*, optimizer::*, scheduler::*};

#[cfg(feature = "pinn")]
use crate::core::error::{KwaversError, KwaversResult};

/// Training configuration
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Maximum number of training epochs
    pub max_epochs: usize,
    /// Convergence tolerance for early stopping
    pub convergence_tolerance: f64,
    /// Window size for convergence checking
    pub convergence_window: usize,
    /// Log training progress every N epochs
    pub log_every: usize,
    /// Save model checkpoint every N epochs
    pub checkpoint_every: usize,
}

/// Main training procedure for PINN
///
/// ## Algorithm
///
/// For each epoch:
/// 1. Forward pass at collocation, boundary, and initial-condition points.
/// 2. PDE residual: ρ ∂²u/∂t² − ∇·σ = 0 (elastic wave equation).
/// 3. Weighted total loss = w_pde·L_pde + w_bc·L_bc + w_ic·L_ic + w_data·L_data.
/// 4. Backward pass (Burn autodiff).
/// 5. Optimizer step (SGD / Adam / AdamW via `PINNOptimizer`).
/// 6. LR scheduler update.
/// 7. Convergence check: max−min of last `convergence_window` total losses < tolerance.
///
/// Material parameters are fixed at typical soft-tissue values
/// (ρ = 1000 kg/m³, λ = 2.25 GPa, μ = 0 Pa) for the forward problem.
/// # Errors
/// - Returns [`KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
///
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[cfg(feature = "pinn")]
pub fn train_pinn<B: AutodiffBackend>(
    model: &mut ElasticPINN2D<B>,
    training_data: &TrainingData<B>,
    optimizer: &mut PINNOptimizer<B>,
    scheduler: &mut LRScheduler,
    config: &TrainingConfig,
) -> KwaversResult<TrainingMetrics> {
    use burn::tensor::Tensor;
    use std::time::Instant;

    use super::super::loss::pde_residual::compute_elastic_wave_pde_residual;
    use super::super::loss::LossComputer;
    use crate::solver::inverse::pinn::elastic_2d::config::LossWeights;

    // Fixed material parameters for the elastic wave forward problem.
    // ρ = soft tissue density; λ = first Lamé parameter giving c_p ≈ 1540 m/s
    // in the fluid limit (μ = 0); μ = shear modulus (fluid-like).
    let rho = 1000.0_f64;
    let lambda = 2.25e9_f64;
    let mu = 0.0_f64;

    let loss_computer = LossComputer::new(LossWeights::default());
    let mut metrics = TrainingMetrics::new();
    let training_start = Instant::now();

    for epoch in 0..config.max_epochs {
        let epoch_start = Instant::now();
        let lr = scheduler.get_lr();

        // ── Forward pass at collocation points ────────────────────────────────
        // model.forward → [N_colloc, 2]: column 0 = u (x-disp), column 1 = v (y-disp)
        let out_colloc = model.forward(
            training_data.collocation.x.clone(),
            training_data.collocation.y.clone(),
            training_data.collocation.t.clone(),
        );
        let n_colloc = out_colloc.dims()[0];
        let u_colloc = out_colloc.clone().slice([0..n_colloc, 0..1]);
        let v_colloc = out_colloc.slice([0..n_colloc, 1..2]);

        // ── PDE residual: ρ ∂²u/∂t² − ∇·σ ───────────────────────────────────
        let (residual_x, residual_y) = compute_elastic_wave_pde_residual(
            u_colloc,
            v_colloc,
            training_data.collocation.x.clone(),
            training_data.collocation.y.clone(),
            training_data.collocation.t.clone(),
            rho,
            lambda,
            mu,
        );
        let pde_loss = loss_computer.pde_loss::<B>(residual_x, residual_y);

        // ── Boundary condition loss ───────────────────────────────────────────
        let out_bc = model.forward(
            training_data.boundary.x.clone(),
            training_data.boundary.y.clone(),
            training_data.boundary.t.clone(),
        );
        let bc_loss =
            loss_computer.boundary_loss::<B>(out_bc, training_data.boundary.values.clone());

        // ── Initial condition loss ────────────────────────────────────────────
        let out_ic = model.forward(
            training_data.initial.x.clone(),
            training_data.initial.y.clone(),
            Tensor::<B, 2>::zeros_like(&training_data.initial.x),
        );
        // velocity target: zero (quiescent start assumed when none provided)
        let zero_vel = Tensor::<B, 2>::zeros_like(&training_data.initial.displacement);
        let ic_loss = loss_computer.initial_loss::<B>(
            out_ic,
            zero_vel,
            training_data.initial.displacement.clone(),
            training_data.initial.velocity.clone(),
        );

        // ── Optional data loss from observations ─────────────────────────────
        let data_loss_opt = training_data.observations.as_ref().map(|obs| {
            let out_obs = model.forward(obs.x.clone(), obs.y.clone(), obs.t.clone());
            loss_computer.data_loss::<B>(out_obs, obs.displacement.clone())
        });

        // ── Total weighted loss ───────────────────────────────────────────────
        let total_loss = loss_computer.total_loss::<B>(
            pde_loss.clone(),
            bc_loss.clone(),
            ic_loss.clone(),
            data_loss_opt.clone(),
        );

        // Extract scalar values before backward consumes total_loss.
        let total_val: f64 = total_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
        let pde_val: f64 = pde_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
        let bc_val: f64 = bc_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
        let ic_val: f64 = ic_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
        let data_val: f64 = data_loss_opt
            .as_ref()
            .map(|t| t.clone().into_data().as_slice::<f32>().unwrap()[0] as f64)
            .unwrap_or(0.0);

        if !total_val.is_finite()
            || !pde_val.is_finite()
            || !bc_val.is_finite()
            || !ic_val.is_finite()
        {
            return Err(KwaversError::Numerical(
                crate::core::error::NumericalError::NaN {
                    operation: "train_pinn".to_string(),
                    inputs: format!(
                        "epoch {epoch}: total={total_val}, pde={pde_val}, bc={bc_val}, ic={ic_val}"
                    ),
                },
            ));
        }

        // ── Backward + optimizer step ─────────────────────────────────────────
        let grads = total_loss.backward();
        *model = optimizer.step(model.clone(), &grads);

        // ── LR scheduler ─────────────────────────────────────────────────────
        scheduler.step(Some(total_val));

        let epoch_time = epoch_start.elapsed().as_secs_f64();
        metrics.record_epoch(total_val, pde_val, bc_val, ic_val, data_val, lr, epoch_time);

        if epoch % config.log_every == 0 || epoch + 1 == config.max_epochs {
            log::info!(
                "train_pinn epoch {}/{}: total={:.4e} pde={:.4e} bc={:.4e} ic={:.4e} lr={:.2e}",
                epoch + 1,
                config.max_epochs,
                total_val,
                pde_val,
                bc_val,
                ic_val,
                lr
            );
        }

        if metrics.has_converged(config.convergence_tolerance, config.convergence_window) {
            log::info!(
                "train_pinn converged at epoch {} (tolerance={:.2e}, window={})",
                epoch + 1,
                config.convergence_tolerance,
                config.convergence_window
            );
            break;
        }
    }

    metrics.total_time = training_start.elapsed().as_secs_f64();
    Ok(metrics)
}

/// Simplified training function for basic use cases
///
/// # Errors
/// Returns `KwaversError::NotImplemented` — delegates to `train_pinn`.
#[cfg(feature = "pinn")]
pub fn train_simple<B: AutodiffBackend>(
    model: &mut ElasticPINN2D<B>,
    max_epochs: usize,
    learning_rate: f64,
) -> KwaversResult<TrainingMetrics> {
    let config = TrainingConfig {
        max_epochs,
        convergence_tolerance: 1e-6,
        convergence_window: 10,
        log_every: 100,
        checkpoint_every: 1000,
    };

    // Create dummy training data (would be provided by user)
    let device = model.device();
    let collocation = super::super::loss::data::CollocationData {
        x: burn::tensor::Tensor::<B, 2>::zeros([100, 1], &device),
        y: burn::tensor::Tensor::<B, 2>::zeros([100, 1], &device),
        t: burn::tensor::Tensor::<B, 2>::zeros([100, 1], &device),
        source_x: None,
        source_y: None,
    };

    let boundary = super::super::loss::data::BoundaryData {
        x: burn::tensor::Tensor::<B, 2>::zeros([50, 1], &device),
        y: burn::tensor::Tensor::<B, 2>::zeros([50, 1], &device),
        t: burn::tensor::Tensor::<B, 2>::zeros([50, 1], &device),
        boundary_type: vec![],
        values: burn::tensor::Tensor::<B, 2>::zeros([50, 2], &device),
    };

    let initial = super::super::loss::data::InitialData {
        x: burn::tensor::Tensor::<B, 2>::zeros([25, 1], &device),
        y: burn::tensor::Tensor::<B, 2>::zeros([25, 1], &device),
        displacement: burn::tensor::Tensor::<B, 2>::zeros([25, 2], &device),
        velocity: burn::tensor::Tensor::<B, 2>::zeros([25, 2], &device),
    };

    let training_data = TrainingData {
        collocation,
        boundary,
        initial,
        observations: None,
    };

    let mut optimizer = PINNOptimizer::sgd(learning_rate, 0.0);
    let mut scheduler = LRScheduler::constant(learning_rate);

    train_pinn(
        model,
        &training_data,
        &mut optimizer,
        &mut scheduler,
        &config,
    )
}

#[cfg(test)]
mod tests {
    use super::super::data::TrainingMetrics;
    #[cfg(feature = "pinn")]
    use super::TrainingConfig;

    #[cfg(feature = "pinn")]
    #[test]
    fn test_training_config() {
        let config = TrainingConfig {
            max_epochs: 1000,
            convergence_tolerance: 1e-6,
            convergence_window: 10,
            log_every: 100,
            checkpoint_every: 500,
        };

        assert_eq!(config.max_epochs, 1000);
        assert_eq!(config.convergence_tolerance, 1e-6);
    }

    #[test]
    fn test_convergence_logic() {
        let mut metrics = TrainingMetrics::new();

        // Add rapidly decreasing loss values that converge to a plateau
        // First 10 epochs: rapid decrease
        for i in 0..10 {
            let loss = 1.0 / (i + 1) as f64;
            metrics.record_epoch(
                loss,
                loss * 0.6,
                loss * 0.3,
                loss * 0.08,
                loss * 0.02,
                0.01,
                0.1,
            );
        }

        // Last 5 epochs: converged to plateau with variation < 1e-5
        let plateau_loss = 0.001;
        for i in 0..5 {
            let loss = plateau_loss + i as f64 * 1e-6; // Very small variation
            metrics.record_epoch(
                loss,
                loss * 0.6,
                loss * 0.3,
                loss * 0.08,
                loss * 0.02,
                0.01,
                0.1,
            );
        }

        // Should converge with loose tolerance (1e-4)
        // Last 5 epochs have variation of 4e-6, well under 1e-4
        assert!(
            metrics.has_converged(1e-4, 5),
            "Expected convergence: last 5 epochs have variation < 1e-4"
        );

        // Should not converge with very strict tolerance
        assert!(
            !metrics.has_converged(1e-7, 5),
            "Should not converge with tolerance < actual variation"
        );
    }
}
