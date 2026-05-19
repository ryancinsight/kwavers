//! Plain data types for `UniversalPINNSolver`.
//!
//! SRP: changes when the public API data shapes change.

use crate::solver::inverse::pinn::ml::physics::{
    InitialConditionSpec, PhysicsLossWeights, PhysicsValidationMetric, PinnBoundaryConditionSpec,
    PinnDomainPhysicsParameters,
};
use std::collections::HashMap;
use std::time::Duration;

/// Universal training configuration for any physics domain
#[derive(Debug, Clone)]
pub struct UniversalTrainingConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub lr_decay: Option<UniversalSolverLrSchedule>,
    pub optimizer: UniversalSolverOptimizerType,
    pub collocation_points: usize,
    pub boundary_points: usize,
    pub initial_points: usize,
    pub adaptive_sampling: bool,
    pub early_stopping: Option<EarlyStoppingConfig>,
    pub batch_size: usize,
    pub gradient_clip: Option<f64>,
    pub physics_weights: PhysicsLossWeights,
}

impl Default for UniversalTrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 1000,
            learning_rate: 0.001,
            lr_decay: Some(UniversalSolverLrSchedule::Exponential { gamma: 0.995 }),
            optimizer: UniversalSolverOptimizerType::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            collocation_points: 1000,
            boundary_points: 200,
            initial_points: 100,
            adaptive_sampling: true,
            early_stopping: Some(EarlyStoppingConfig {
                patience: 50,
                min_delta: 1e-6,
                restore_best_weights: true,
            }),
            batch_size: 32,
            gradient_clip: Some(1.0),
            physics_weights: PhysicsLossWeights::default(),
        }
    }
}

/// Learning rate decay schedules
#[derive(Debug, Clone)]
pub enum UniversalSolverLrSchedule {
    /// Exponential decay: lr *= gamma^epoch
    Exponential { gamma: f64 },
    /// Step decay: lr *= gamma every step_size epochs
    Step { gamma: f64, step_size: usize },
    /// Cosine annealing
    Cosine { t_max: usize, eta_min: f64 },
}

/// Optimizer types for PINN training
#[derive(Debug, Clone)]
pub enum UniversalSolverOptimizerType {
    Adam {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    LBFGS {
        history_size: usize,
        line_search_method: LineSearchMethod,
    },
    SGD {
        momentum: f64,
    },
}

/// Line search methods for L-BFGS
#[derive(Debug, Clone)]
pub enum LineSearchMethod {
    Backtracking { alpha: f64, beta: f64 },
    StrongWolfe { c1: f64, c2: f64 },
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    pub patience: usize,
    pub min_delta: f64,
    pub restore_best_weights: bool,
}

/// Universal solver statistics and performance metrics
#[derive(Debug, Clone)]
pub struct UniversalSolverStats {
    pub training_time: Duration,
    pub final_losses: HashMap<String, f64>,
    pub final_loss: f64,
    pub loss_history: Vec<HashMap<String, f64>>,
    pub physics_metrics: Vec<PhysicsValidationMetric>,
    pub convergence_info: UniversalSolverConvergenceInfo,
    pub memory_stats: Option<UniversalSolverMemoryStats>,
}

impl Default for UniversalSolverStats {
    fn default() -> Self {
        Self {
            training_time: Duration::default(),
            final_losses: HashMap::new(),
            final_loss: 0.0,
            loss_history: Vec::new(),
            physics_metrics: Vec::new(),
            convergence_info: UniversalSolverConvergenceInfo {
                converged: false,
                final_epoch: 0,
                best_loss: f64::INFINITY,
                best_epoch: 0,
                loss_reduction_ratio: 1.0,
            },
            memory_stats: None,
        }
    }
}

/// Result of training multiple physics domains
#[derive(Debug, Clone)]
pub struct MultiDomainTrainingResult {
    pub total_loss: f64,
    pub training_time: Duration,
    pub domain_stats: HashMap<String, UniversalSolverStats>,
}

/// Convergence information
#[derive(Debug, Clone)]
pub struct UniversalSolverConvergenceInfo {
    pub converged: bool,
    pub final_epoch: usize,
    pub best_loss: f64,
    pub best_epoch: usize,
    pub loss_reduction_ratio: f64,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct UniversalSolverMemoryStats {
    pub peak_gpu_memory_mb: f64,
    pub peak_cpu_memory_mb: f64,
    pub final_gpu_memory_mb: f64,
}

/// Physics solution containing trained model and metadata
#[derive(Debug)]
pub struct PhysicsSolution<B: burn::tensor::backend::AutodiffBackend> {
    pub model: crate::solver::inverse::pinn::ml::BurnPINN2DWave<B>,
    pub config: UniversalTrainingConfig,
    pub stats: UniversalSolverStats,
    pub domain_info: UniversalSolverDomainInfo,
}

/// Domain information for solution metadata
#[derive(Debug, Clone)]
pub struct UniversalSolverDomainInfo {
    pub domain_name: String,
    pub physics_params: PinnDomainPhysicsParameters,
    pub boundary_conditions: Vec<PinnBoundaryConditionSpec>,
    pub initial_conditions: Vec<InitialConditionSpec>,
}

/// Geometry specification for 2D domains
#[derive(Debug, Clone)]
pub struct UniversalSolverGeometry2D {
    pub bounds: [f64; 4],
    pub features: Vec<GeometricFeature>,
}

impl UniversalSolverGeometry2D {
    pub fn rectangle(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        Self {
            bounds: [x_min, x_max, y_min, y_max],
            features: Vec::new(),
        }
    }

    pub fn with_circle_obstacle(mut self, center: (f64, f64), radius: f64) -> Self {
        self.features
            .push(GeometricFeature::Circle { center, radius });
        self
    }

    pub fn with_rectangle_obstacle(
        mut self,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
    ) -> Self {
        self.features.push(GeometricFeature::Rectangle {
            x_min,
            x_max,
            y_min,
            y_max,
        });
        self
    }
}

/// Geometric features in the domain
#[derive(Debug, Clone)]
pub enum GeometricFeature {
    Circle {
        center: (f64, f64),
        radius: f64,
    },
    Rectangle {
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
    },
    Interface {
        points: Vec<(f64, f64)>,
    },
}
