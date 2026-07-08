//! Main training loop and coordination for PINN optimization
//!
//! This module implements the high-level training procedure that coordinates
//! data loading, loss computation, optimization, and convergence checking.

use coeus_autograd::Var;

use super::super::model::ElasticPINN2D;
use super::{data::*, optimizer::*, scheduler::*};

use kwavers_core::error::{KwaversError, KwaversResult};

/// Training configuration
#[derive(Debug, Clone)]
pub struct ElasticPinnLoopConfig {
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
/// 4. Backward pass (`coeus_autograd`).
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
pub fn train_pinn<B>(
    model: &mut ElasticPINN2D<B>,
    training_data: &TrainingData<B>,
    optimizer: &mut PINNOptimizer<B>,
    scheduler: &mut LRScheduler,
    config: &ElasticPinnLoopConfig,
) -> KwaversResult<ElasticPinnTrainingMetrics>
where
    B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    use std::time::Instant;

    use super::super::loss::pde_residual::compute_elastic_wave_pde_residual;
    use super::super::loss::LossComputer;
    use crate::inverse::pinn::elastic_2d::config::LossWeights;

    // Fixed material parameters for the elastic wave forward problem.
    // ρ = water-equivalent density (fluid limit μ=0); λ gives c_p ≈ 1500 m/s.
    let rho = kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
    let lambda = 2.25e9_f64;
    let mu = 0.0_f64;

    let loss_computer = LossComputer::new(LossWeights::default());
    let mut metrics = ElasticPinnTrainingMetrics::new();
    let training_start = Instant::now();

    for epoch in 0..config.max_epochs {
        let epoch_start = Instant::now();
        let lr = scheduler.get_lr();

        for p in model.parameters() {
            p.zero_grad();
        }

        // ── PDE residual: ρ ∂²u/∂t² − ∇·σ ───────────────────────────────────
        let (residual_x, residual_y) = compute_elastic_wave_pde_residual(
            model,
            &training_data.collocation.x,
            &training_data.collocation.y,
            &training_data.collocation.t,
            rho,
            lambda,
            mu,
        )?;
        let pde_loss = loss_computer.pde_loss::<B>(&residual_x, &residual_y);

        // ── Boundary condition loss ───────────────────────────────────────────
        let out_bc = model.forward(
            &training_data.boundary.x,
            &training_data.boundary.y,
            &training_data.boundary.t,
        );
        let bc_loss = loss_computer.boundary_loss::<B>(&out_bc, &training_data.boundary.values);

        // ── Initial condition loss ────────────────────────────────────────────
        let backend = B::default();
        let zero_t = Var::new(
            coeus_tensor::Tensor::zeros_on(training_data.initial.x.tensor.shape(), &backend),
            false,
        );
        let out_ic = model.forward(&training_data.initial.x, &training_data.initial.y, &zero_t);
        // velocity target: zero (quiescent start assumed when none provided)
        let zero_vel = Var::new(
            coeus_tensor::Tensor::zeros_on(
                training_data.initial.displacement.tensor.shape(),
                &backend,
            ),
            false,
        );
        let ic_loss = loss_computer.initial_loss::<B>(
            &out_ic,
            &zero_vel,
            &training_data.initial.displacement,
            &training_data.initial.velocity,
        );

        // ── Optional data loss from observations ─────────────────────────────
        let data_loss_opt = training_data.observations.as_ref().map(|obs| {
            let out_obs = model.forward(&obs.x, &obs.y, &obs.t);
            loss_computer.data_loss::<B>(&out_obs, &obs.displacement)
        });

        // ── Total weighted loss ───────────────────────────────────────────────
        let total_loss =
            loss_computer.total_loss::<B>(&pde_loss, &bc_loss, &ic_loss, data_loss_opt.as_ref());

        // Extract scalar values before backward.
        let total_val = total_loss.tensor.as_slice()[0] as f64;
        let pde_val = pde_loss.tensor.as_slice()[0] as f64;
        let bc_val = bc_loss.tensor.as_slice()[0] as f64;
        let ic_val = ic_loss.tensor.as_slice()[0] as f64;
        let data_val = data_loss_opt
            .as_ref()
            .map(|t| t.tensor.as_slice()[0] as f64)
            .unwrap_or(0.0);

        if !total_val.is_finite()
            || !pde_val.is_finite()
            || !bc_val.is_finite()
            || !ic_val.is_finite()
        {
            return Err(KwaversError::Numerical(
                kwavers_core::error::NumericalError::NaN {
                    operation: "train_pinn".to_string(),
                    inputs: format!(
                        "epoch {epoch}: total={total_val}, pde={pde_val}, bc={bc_val}, ic={ic_val}"
                    ),
                },
            ));
        }

        // ── Backward + optimizer step ─────────────────────────────────────────
        total_loss.backward();
        optimizer.step(model);

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

/// Simplified training entry point for basic use cases.
///
/// Constructs zero-initialised collocation/boundary/initial datasets and a
/// constant-rate SGD optimiser, then delegates to [`train_pinn`]. Intended for
/// smoke tests and examples; supply real [`TrainingData`] via [`train_pinn`]
/// for production runs.
///
/// # Errors
/// Propagates any [`KwaversError`] returned by [`train_pinn`] (e.g. a tensor /
/// autodiff backend failure during an epoch).
pub fn train_simple<B>(
    model: &mut ElasticPINN2D<B>,
    max_epochs: usize,
    learning_rate: f64,
) -> KwaversResult<ElasticPinnTrainingMetrics>
where
    B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    let config = ElasticPinnLoopConfig {
        max_epochs,
        convergence_tolerance: 1e-6,
        convergence_window: 10,
        log_every: 100,
        checkpoint_every: 1000,
    };

    // Create dummy training data (would be provided by user)
    let backend = B::default();
    let zeros =
        |shape: Vec<usize>| Var::new(coeus_tensor::Tensor::zeros_on(shape, &backend), false);

    let collocation = super::super::loss::data::CollocationData {
        x: zeros(vec![100, 1]),
        y: zeros(vec![100, 1]),
        t: zeros(vec![100, 1]),
        source_x: None,
        source_y: None,
    };

    let boundary = super::super::loss::data::BoundaryData {
        x: zeros(vec![50, 1]),
        y: zeros(vec![50, 1]),
        t: zeros(vec![50, 1]),
        boundary_type: vec![],
        values: zeros(vec![50, 2]),
    };

    let initial = super::super::loss::data::InitialData {
        x: zeros(vec![25, 1]),
        y: zeros(vec![25, 1]),
        displacement: zeros(vec![25, 2]),
        velocity: zeros(vec![25, 2]),
    };

    let training_data = TrainingData {
        collocation,
        boundary,
        initial,
        observations: None,
    };

    let mut optimizer = PINNOptimizer::sgd(model, learning_rate, 0.0);
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
    use super::super::data::ElasticPinnTrainingMetrics;
    use super::ElasticPinnLoopConfig;

    #[test]
    fn test_training_config() {
        let config = ElasticPinnLoopConfig {
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
        let mut metrics = ElasticPinnTrainingMetrics::new();

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

    #[test]
    fn smoke_train_simple_runs_and_produces_finite_metrics() {
        use crate::inverse::pinn::elastic_2d::model::ElasticPINN2D;
        use crate::inverse::pinn::elastic_2d::training::train_simple;
        use crate::inverse::pinn::elastic_2d::Config;
        type B = coeus_core::MoiraiBackend;

        let config = Config {
            hidden_layers: vec![4],
            ..Config::default()
        };
        let mut model = ElasticPINN2D::<B>::new(&config).unwrap();
        let metrics = train_simple(&mut model, 3, 1e-3).unwrap();
        assert_eq!(metrics.epochs_completed, 3);
        assert!(metrics.final_loss().unwrap().is_finite());
    }
}
