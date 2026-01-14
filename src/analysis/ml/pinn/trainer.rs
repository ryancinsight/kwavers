use crate::analysis::ml::pinn::burn_wave_equation_1d::{BurnPINNConfig, BurnPINNTrainer};
use crate::analysis::ml::pinn::burn_wave_equation_2d::{
    BurnPINN2DConfig, BurnPINN2DTrainer, Geometry2D,
};
use crate::analysis::ml::pinn::wave_equation_1d::TrainingMetrics;
use crate::core::error::{KwaversError, KwaversResult};
use burn::backend::{Autodiff, NdArray};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use tokio::sync::mpsc;

use serde::{Deserialize, Serialize};

/// Configuration for PINN training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PINNConfig {
    /// Physics domain (e.g., "acoustic_wave")
    pub physics_domain: String,
    /// Geometry specification
    pub geometry: Geometry,
    /// Physics parameters
    pub physics_params: PhysicsParams,
    /// Training configuration
    pub training_config: TrainingConfig,
    /// Whether to use GPU acceleration
    pub use_gpu: bool,
}

/// Geometry specification for PINN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Geometry {
    pub bounds: Vec<f64>,
    pub obstacles: Vec<Obstacle>,
    pub boundary_conditions: Vec<BoundaryCondition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Obstacle {
    pub shape: String,
    pub center: Vec<f64>,
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryCondition {
    pub boundary: String,
    pub condition_type: String,
    pub value: f64,
}

/// Physics parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsParams {
    pub material_properties: HashMap<String, f64>,
    pub boundary_values: HashMap<String, f64>,
    pub initial_values: HashMap<String, f64>,
    pub domain_params: HashMap<String, f64>,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub collocation_points: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub hidden_layers: Vec<usize>,
    pub adaptive_sampling: bool,
    pub use_gpu: bool,
}

/// Result of PINN training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    pub metrics: TrainingMetrics,
    // Add other fields as needed, e.g., model weights
}

/// PINN Trainer
#[derive(Debug)]
pub struct PINNTrainer {
    config: PINNConfig,
}

impl PINNTrainer {
    /// Create a new PINN trainer
    pub fn new(config: PINNConfig) -> KwaversResult<Self> {
        Ok(Self { config })
    }

    /// Train the PINN model with progress updates
    pub async fn train_with_progress(
        &mut self,
        progress_sender: mpsc::Sender<crate::api::TrainingProgress>,
    ) -> KwaversResult<TrainingResult> {
        // Validation
        if self.config.physics_domain != "acoustic_wave" {
            return Err(KwaversError::InvalidInput(format!(
                "Unsupported physics domain: {}",
                self.config.physics_domain
            )));
        }

        let config = self.config.clone();

        // Offload training to blocking thread
        let result = tokio::task::spawn_blocking(move || {
            let is_2d = config.geometry.bounds.len() >= 4;

            if is_2d {
                train_2d(config, progress_sender)
            } else {
                train_1d(config, progress_sender)
            }
        })
        .await
        .map_err(|e| KwaversError::InternalError(e.to_string()))??;

        Ok(result)
    }
}

fn train_1d(
    config: PINNConfig,
    sender: mpsc::Sender<crate::api::TrainingProgress>,
) -> KwaversResult<TrainingResult> {
    type Backend = Autodiff<NdArray<f32>>;
    let device = Default::default();

    let burn_config = BurnPINNConfig {
        hidden_layers: config.training_config.hidden_layers.clone(),
        learning_rate: config.training_config.learning_rate,
        loss_weights: Default::default(),
        num_collocation_points: config.training_config.collocation_points,
    };

    let mut trainer = BurnPINNTrainer::<Backend>::new(burn_config, &device)?;

    let n = config.training_config.batch_size;
    let x_min = config.geometry.bounds.first().copied().unwrap_or(-1.0);
    let x_max = config.geometry.bounds.get(1).copied().unwrap_or(1.0);
    let x_data = Array1::linspace(x_min, x_max, n);
    let t_data = Array1::linspace(0.0, 1.0, n);
    let u_data = Array2::zeros((n, 1));
    let wave_speed = config
        .physics_params
        .material_properties
        .get("wave_speed")
        .cloned()
        .unwrap_or(1500.0);

    let start_time = std::time::Instant::now();

    let metrics = trainer.train_with_callback(
        &x_data,
        &t_data,
        &u_data,
        wave_speed,
        &device,
        config.training_config.epochs,
        |epoch, metrics| {
            let progress = crate::api::TrainingProgress {
                current_epoch: epoch + 1,
                total_epochs: config.training_config.epochs,
                current_loss: metrics.total_loss.last().cloned().unwrap_or(0.0),
                best_loss: 0.0, // simplified
                elapsed_seconds: start_time.elapsed().as_secs(),
                estimated_remaining: 0,
            };
            sender.blocking_send(progress).is_ok()
        },
    )?;

    Ok(TrainingResult {
        metrics: TrainingMetrics {
            total_loss: metrics.total_loss,
            data_loss: metrics.data_loss,
            pde_loss: metrics.pde_loss,
            bc_loss: metrics.bc_loss,
            training_time_secs: metrics.training_time_secs,
            epochs_completed: metrics.epochs_completed,
        },
    })
}

fn train_2d(
    config: PINNConfig,
    sender: mpsc::Sender<crate::api::TrainingProgress>,
) -> KwaversResult<TrainingResult> {
    type Backend = Autodiff<NdArray<f32>>;
    let device = Default::default();

    let burn_config = BurnPINN2DConfig {
        hidden_layers: config.training_config.hidden_layers.clone(),
        learning_rate: config.training_config.learning_rate,
        loss_weights: Default::default(),
        num_collocation_points: config.training_config.collocation_points,
        boundary_condition: Default::default(),
    };

    let x_min = config.geometry.bounds.first().copied().unwrap_or(-1.0);
    let x_max = config.geometry.bounds.get(1).copied().unwrap_or(1.0);
    let y_min = config.geometry.bounds.get(2).copied().unwrap_or(-1.0);
    let y_max = config.geometry.bounds.get(3).copied().unwrap_or(1.0);

    let geometry = Geometry2D::new(x_min, x_max, y_min, y_max);

    let mut trainer = BurnPINN2DTrainer::<Backend>::new_trainer(burn_config, geometry, &device)?;

    let n = config.training_config.batch_size;
    let x_data = Array1::linspace(x_min, x_max, n);
    let y_data = Array1::linspace(y_min, y_max, n);
    let t_data = Array1::linspace(0.0, 1.0, n);
    let u_data = Array2::zeros((n, 1));
    let wave_speed = config
        .physics_params
        .material_properties
        .get("wave_speed")
        .cloned()
        .unwrap_or(1500.0);

    let start_time = std::time::Instant::now();

    let metrics = trainer.train_with_callback(
        &x_data,
        &y_data,
        &t_data,
        &u_data,
        wave_speed,
        trainer.pinn.config(),
        &device,
        config.training_config.epochs,
        |epoch, metrics| {
            let progress = crate::api::TrainingProgress {
                current_epoch: epoch + 1,
                total_epochs: config.training_config.epochs,
                current_loss: metrics.total_loss.last().cloned().unwrap_or(0.0),
                best_loss: 0.0, // simplified
                elapsed_seconds: start_time.elapsed().as_secs(),
                estimated_remaining: 0,
            };
            sender.blocking_send(progress).is_ok()
        },
    )?;

    Ok(TrainingResult {
        metrics: TrainingMetrics {
            total_loss: metrics.total_loss,
            data_loss: metrics.data_loss,
            pde_loss: metrics.pde_loss,
            bc_loss: metrics.bc_loss,
            training_time_secs: metrics.training_time_secs,
            epochs_completed: metrics.epochs_completed,
        },
    })
}
