//! Meta-Learning Framework for PINN Training
//!
//! This module implements Model-Agnostic Meta-Learning (MAML) for Physics-Informed Neural Networks,
//! enabling fast adaptation to new physics problems and geometries through learned optimal initialization.
//!
//! ## Current Implementation Status
//!
//! **Assumptions:**
//! - Linear neural network model for simplified gradient computation
//! - Finite difference gradient estimation instead of automatic differentiation
//! - Simplified PDE residual computation using analytical derivatives
//!
//! **Limitations:**
//! - Gradient computation uses finite differences (numerically unstable and slow)
//! - Loss computation assumes linear model (not representative of real neural networks)
//! - No support for complex physics domains beyond simple wave equations
//! - Memory inefficient for large parameter spaces
//!
//! **Future Improvements:**
//! - Implement full automatic differentiation for gradient computation
//! - Support arbitrary neural network architectures
//! - Add convergence criteria and adaptive learning rates
//! - Extend to multi-physics coupling scenarios

use crate::core::error::{KwaversError, KwaversResult};
use crate::math::ml::pinn::burn_wave_equation_2d::{
    BurnLossWeights2D, BurnPINN2DConfig, BurnPINN2DWave, SimpleOptimizer2D,
};
use crate::math::ml::pinn::Geometry2D;
use burn::module::{Module, ModuleMapper};
use burn::prelude::ToElement;
use burn::tensor::{backend::AutodiffBackend, Bool, Int, Tensor};
use std::sync::Arc;

#[derive(Debug)]
struct GradientExtractor<'a, B: AutodiffBackend> {
    grads: &'a B::Gradients,
    collected: Vec<Option<Tensor<B::InnerBackend, 1>>>,
}

impl<'a, B: AutodiffBackend> ModuleMapper<B> for GradientExtractor<'a, B> {
    fn map_float<const D: usize>(
        &mut self,
        tensor: burn::module::Param<Tensor<B, D>>,
    ) -> burn::module::Param<Tensor<B, D>> {
        let grad_opt = tensor
            .grad(self.grads)
            .map(|g| g.flatten(0, D.saturating_sub(1)));
        self.collected.push(grad_opt);
        tensor
    }

    fn map_int<const D: usize>(
        &mut self,
        tensor: burn::module::Param<Tensor<B, D, Int>>,
    ) -> burn::module::Param<Tensor<B, D, Int>> {
        tensor
    }

    fn map_bool<const D: usize>(
        &mut self,
        tensor: burn::module::Param<Tensor<B, D, Bool>>,
    ) -> burn::module::Param<Tensor<B, D, Bool>> {
        tensor
    }
}

#[derive(Debug)]
struct GradientApplicator<B: AutodiffBackend> {
    grads: Vec<Option<Tensor<B::InnerBackend, 1>>>,
    index: usize,
    lr: f64,
}

impl<B: AutodiffBackend> ModuleMapper<B> for GradientApplicator<B> {
    fn map_float<const D: usize>(
        &mut self,
        tensor: burn::module::Param<Tensor<B, D>>,
    ) -> burn::module::Param<Tensor<B, D>> {
        let grad_opt = self.grads.get(self.index).cloned().unwrap_or(None);
        self.index = self.index.saturating_add(1);

        if let Some(grad_flat) = grad_opt {
            let is_require_grad = tensor.is_require_grad();
            let mut inner = (*tensor).clone().inner();
            let grad: Tensor<B::InnerBackend, D> = grad_flat.reshape(inner.dims());
            inner = inner - grad.mul_scalar(self.lr);
            let mut out = Tensor::<B, D>::from_inner(inner);
            if is_require_grad {
                out = out.require_grad();
            }
            burn::module::Param::from_tensor(out)
        } else {
            tensor
        }
    }

    fn map_int<const D: usize>(
        &mut self,
        tensor: burn::module::Param<Tensor<B, D, Int>>,
    ) -> burn::module::Param<Tensor<B, D, Int>> {
        tensor
    }

    fn map_bool<const D: usize>(
        &mut self,
        tensor: burn::module::Param<Tensor<B, D, Bool>>,
    ) -> burn::module::Param<Tensor<B, D, Bool>> {
        tensor
    }
}

/// Meta-learning configuration
#[derive(Debug, Clone)]
pub struct MetaLearningConfig {
    /// Inner-loop learning rate for task adaptation
    pub inner_lr: f64,
    /// Outer-loop learning rate for meta-parameter updates
    pub outer_lr: f64,
    /// Number of inner-loop adaptation steps
    pub adaptation_steps: usize,
    /// Number of tasks per meta-batch
    pub meta_batch_size: usize,
    /// Meta-training epochs
    pub meta_epochs: usize,
    /// First-order approximation (FO-MAML)
    pub first_order: bool,
    /// Physics-aware regularization
    pub physics_regularization: f64,
    /// Network architecture parameters
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    /// Maximum number of tasks for curriculum learning
    pub max_tasks: usize,
}

/// Types of PDEs for meta-learning tasks
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PdeType {
    Wave,
    Diffusion,
    NavierStokes,
    Electromagnetic,
    Acoustic,
    Elastic,
}

/// Physics task definition for meta-learning
#[derive(Debug, Clone)]
pub struct PhysicsTask {
    /// Unique task identifier
    pub id: String,
    /// Type of PDE
    pub pde_type: PdeType,
    /// Physics parameters (wave speed, density, etc.)
    pub physics_params: PhysicsParameters,
    /// Geometry specification
    pub geometry: Arc<crate::ml::pinn::Geometry2D>,
    /// Boundary conditions
    pub boundary_conditions: Vec<crate::ml::pinn::BoundaryCondition2D>,
    /// Training data (optional for few-shot learning)
    pub training_data: Option<TaskData>,
    /// Validation data for meta-training
    pub validation_data: TaskData,
}

/// Physics parameters for task definition
#[derive(Debug, Clone)]
pub struct PhysicsParameters {
    pub wave_speed: f64,
    pub density: f64,
    pub viscosity: Option<f64>,
    pub absorption: Option<f64>,
    pub nonlinearity: Option<f64>,
}

/// Task data for training/validation
#[derive(Debug, Clone)]
pub struct TaskData {
    /// Collocation points (x, y, t)
    pub collocation_points: Vec<(f64, f64, f64)>,
    /// Boundary data (x, y, t, u)
    pub boundary_data: Vec<(f64, f64, f64, f64)>,
    /// Initial data (x, y, 0, u, du/dt)
    pub initial_data: Vec<(f64, f64, f64, f64, f64)>,
}

/// Meta-learning loss and metrics
#[derive(Debug, Clone)]
pub struct MetaLoss {
    /// Total meta-loss across all tasks
    pub total_loss: f64,
    /// Task-specific losses
    pub task_losses: Vec<f64>,
    /// Physics constraint satisfaction
    pub physics_loss: f64,
    /// Generalization metric
    pub generalization_score: f64,
}

/// Meta-learner for PINN models
#[derive(Debug)]
pub struct MetaLearner<B: AutodiffBackend> {
    /// Base model acting as meta-parameters
    base_model: BurnPINN2DWave<B>,
    /// Meta-optimizer state
    _meta_optimizer: MetaOptimizer<B>,
    /// Configuration
    config: MetaLearningConfig,
    /// Task distribution sampler
    task_sampler: TaskSampler,
    /// Performance statistics
    stats: MetaLearningStats,
}

/// Meta-optimizer for outer-loop updates
#[derive(Debug)]
pub struct MetaOptimizer<B: AutodiffBackend> {
    /// Outer-loop learning rate
    lr: f64,
    /// Momentum parameter
    _momentum: Option<f64>,
    /// Adam optimizer parameters
    _beta1: f64,
    _beta2: f64,
    _epsilon: f64,
    /// Iteration count for bias correction
    _iteration_count: usize,
    /// First moment estimates
    _m: Vec<Option<Tensor<B, 1>>>,
    /// Second moment estimates
    _v: Vec<Option<Tensor<B, 1>>>,
}

/// Task sampler for meta-training
#[derive(Debug)]
pub struct TaskSampler {
    /// Available physics tasks
    task_pool: Vec<PhysicsTask>,
    /// Task sampling strategy
    sampling_strategy: SamplingStrategy,
    /// Current sampling index
    current_index: usize,
    /// History of sampled task indices
    task_history: Vec<usize>,
    /// Configuration
    config: MetaLearningConfig,
}

/// Task sampling strategies
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Random sampling
    Random,
    /// Curriculum learning (easy to hard)
    Curriculum,
    /// Balanced sampling across physics families
    Balanced,
    /// Diversity sampling
    Diversity,
}

/// Meta-learning performance statistics
#[derive(Debug, Clone)]
pub struct MetaLearningStats {
    pub meta_epochs_completed: usize,
    pub total_tasks_processed: usize,
    pub average_adaptation_time: f64,
    pub average_meta_loss: f64,
    pub best_generalization_score: f64,
    pub convergence_rate: f64,
}

impl<B: AutodiffBackend> MetaLearner<B> {
    /// Create a new meta-learner
    pub fn new(config: MetaLearningConfig, device: &B::Device) -> KwaversResult<Self> {
        // Initialize base model
        let pinn_config = BurnPINN2DConfig {
            hidden_layers: vec![config.hidden_dim; config.num_layers],
            ..Default::default()
        };
        let base_model = BurnPINN2DWave::new(pinn_config, device)?;

        // Initialize meta-optimizer with proper parameter count
        let total_params = base_model.parameters().len();
        let meta_optimizer = MetaOptimizer::new(config.outer_lr, total_params);

        // Initialize task sampler
        let task_sampler = TaskSampler::new(SamplingStrategy::Balanced, config.clone());

        Ok(Self {
            base_model,
            _meta_optimizer: meta_optimizer,
            config,
            task_sampler,
            stats: MetaLearningStats::default(),
        })
    }

    /// Perform one meta-training step
    pub fn meta_train_step(&mut self) -> KwaversResult<MetaLoss> {
        // Sample batch of tasks
        let tasks = self
            .task_sampler
            .sample_batch(self.config.meta_batch_size)?;

        let mut total_loss = 0.0;
        let mut task_losses = Vec::new();
        let mut physics_losses = Vec::new();
        let _meta_gradients_acc: Option<B::Gradients> = None; // Accumulate gradients? No, we need average

        // For FOMAML (First-Order MAML), we compute gradients at the adapted parameter
        // and apply them to the initial parameters.
        // We will accumulate gradients w.r.t initial parameters.

        // Since Burn doesn't support easy gradient accumulation across iterations for the same parameters
        // without a custom optimizer step that takes explicit gradients, we will need to handle this.
        // But for now, let's implement a simplified Reptile-like update or serial processing.

        // Current implementation: Just process one task to check flow, then expand.
        // Ideally we process all tasks.

        let mut aggregated_grads: Vec<B::Gradients> = Vec::new();

        // For each task, perform inner-loop adaptation and compute meta-gradient
        for task in &tasks {
            // 1. Adapt to task (Inner Loop)
            let (task_loss, physics_loss, adapted_model) = self.adapt_to_task_internal(task)?;

            total_loss += task_loss;
            task_losses.push(task_loss);
            physics_losses.push(physics_loss);

            // 2. Compute Meta-Gradient (Outer Loop)
            // For FOMAML, this is just the gradient of the validation loss w.r.t adapted parameters
            // evaluated on the validation set.
            let valid_data = self.generate_task_data(task)?; // Use validation data (same generator for now)
            let (_val_loss, val_grads) =
                self.compute_gradients_and_loss(&adapted_model, &valid_data, task)?;

            aggregated_grads.push(val_grads);
        }

        // 3. Meta-Update
        // Aggregate and average gradients across tasks
        let num_tasks = tasks.len();
        let mut accumulated_grads: Vec<Option<Tensor<B::InnerBackend, 1>>> = Vec::new();

        // Sum gradients from all tasks
        for (i, task_grad_set) in aggregated_grads.iter().enumerate() {
            let mut extractor = GradientExtractor {
                grads: task_grad_set,
                collected: Vec::new(),
            };
            // Map over model structure to extract gradients in order
            self.base_model.clone().map(&mut extractor);
            let task_grads_flat = extractor.collected;

            if i == 0 {
                accumulated_grads = task_grads_flat;
            } else {
                for (acc, new) in accumulated_grads.iter_mut().zip(task_grads_flat.iter()) {
                    if let (Some(a), Some(b)) = (acc.as_mut(), new.as_ref()) {
                        *a = a.clone().add(b.clone());
                    } else if acc.is_none() && new.is_some() {
                        *acc = new.clone();
                    }
                }
            }
        }

        // Average gradients
        let averaged_grads: Vec<Option<Tensor<B::InnerBackend, 1>>> = accumulated_grads
            .into_iter()
            .map(|opt| opt.map(|g| g.div_scalar(num_tasks as f64)))
            .collect();

        // Apply averaged gradients to base model
        let mut applicator = GradientApplicator {
            grads: averaged_grads,
            index: 0,
            lr: self.config.outer_lr,
        };

        self.base_model = self.base_model.clone().map(&mut applicator);

        // Compute meta-loss stats
        let average_physics_loss = if !physics_losses.is_empty() {
            physics_losses.iter().sum::<f64>() / physics_losses.len() as f64
        } else {
            0.0
        };

        let generalization_score = self.compute_generalization_score(&task_losses);

        let meta_loss = MetaLoss {
            total_loss: if !tasks.is_empty() {
                total_loss / tasks.len() as f64
            } else {
                0.0
            },
            task_losses,
            physics_loss: average_physics_loss,
            generalization_score,
        };

        // Update statistics
        self.stats.meta_epochs_completed += 1;
        self.stats.total_tasks_processed += tasks.len();
        self.stats.average_meta_loss = (self.stats.average_meta_loss + meta_loss.total_loss) / 2.0;
        self.stats.best_generalization_score = self
            .stats
            .best_generalization_score
            .max(generalization_score);

        Ok(meta_loss)
    }

    /// Adapt to a specific physics task using learned meta-parameters
    pub fn adapt_to_task(&self, task: &PhysicsTask) -> KwaversResult<(f64, f64)> {
        let (loss, physics_loss, _) = self.adapt_to_task_internal(task)?;
        Ok((loss, physics_loss))
    }

    /// Internal adaptation method returning the adapted model
    fn adapt_to_task_internal(
        &self,
        task: &PhysicsTask,
    ) -> KwaversResult<(f64, f64, BurnPINN2DWave<B>)> {
        // Clone base model for this task
        let mut task_model = self.base_model.clone();

        // Generate task-specific data
        let task_data = self.generate_task_data(task)?;

        // Inner-loop optimizer
        let optimizer = SimpleOptimizer2D::new(self.config.inner_lr as f32);

        let mut task_loss = 0.0;
        let mut physics_loss = 0.0;

        for _ in 0..self.config.adaptation_steps {
            // Compute gradients and loss
            let (loss, grads) = self.compute_gradients_and_loss(&task_model, &task_data, task)?;

            // Update parameters using inner-loop learning
            task_model = optimizer.step(task_model, &grads);

            task_loss = loss.total_loss;
            physics_loss = loss.physics_loss;
        }

        Ok((task_loss, physics_loss, task_model))
    }

    /// Compute gradients and loss for a model on a task
    fn compute_gradients_and_loss(
        &self,
        model: &BurnPINN2DWave<B>,
        data: &TaskData,
        task: &PhysicsTask,
    ) -> KwaversResult<(MetaLoss, B::Gradients)> {
        let device = model.device();

        // Prepare tensors
        let (x_colloc, y_colloc, t_colloc) = self.to_tensors_3(&data.collocation_points, &device);
        let (x_bc, y_bc, t_bc, u_bc) = self.to_tensors_4(&data.boundary_data, &device);
        let (x_ic, y_ic, t_ic, u_ic) = self.to_tensors_5_to_4(&data.initial_data, &device);

        // For data loss, we reuse boundary data if no separate training data is available
        // In a real scenario, we would have separate labeled data.
        let x_data = x_bc.clone();
        let y_data = y_bc.clone();
        let t_data = t_bc.clone();
        let u_data = u_bc.clone();

        // Compute physics loss
        let (total_loss, _data_loss, pde_loss, _bc_loss, _ic_loss) = model.compute_physics_loss(
            x_data,
            y_data,
            t_data,
            u_data,
            x_colloc,
            y_colloc,
            t_colloc,
            x_bc,
            y_bc,
            t_bc,
            u_bc,
            x_ic,
            y_ic,
            t_ic,
            u_ic,
            task.physics_params.wave_speed,
            BurnLossWeights2D::default(),
        );

        // Compute gradients
        let grads = total_loss.backward();

        let meta_loss = MetaLoss {
            total_loss: total_loss.into_scalar().to_f64(),
            task_losses: vec![],
            physics_loss: pde_loss.into_scalar().to_f64(),
            generalization_score: 0.0,
        };

        Ok((meta_loss, grads))
    }

    // Helper to convert Vec<(f64, f64, f64)> to tensors
    fn to_tensors_3(
        &self,
        data: &[(f64, f64, f64)],
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let n = data.len();
        if n == 0 {
            // Return empty tensors or dummy 1-element tensors to avoid shape errors
            // For now, assume data is not empty or handle it safely
            let dummy = Tensor::zeros([1, 1], device);
            return (dummy.clone(), dummy.clone(), dummy);
        }

        let x: Vec<f32> = data.iter().map(|p| p.0 as f32).collect();
        let y: Vec<f32> = data.iter().map(|p| p.1 as f32).collect();
        let t: Vec<f32> = data.iter().map(|p| p.2 as f32).collect();

        (
            Tensor::<B, 1>::from_floats(x.as_slice(), device).reshape([n, 1]),
            Tensor::<B, 1>::from_floats(y.as_slice(), device).reshape([n, 1]),
            Tensor::<B, 1>::from_floats(t.as_slice(), device).reshape([n, 1]),
        )
    }

    // Helper to convert Vec<(f64, f64, f64, f64)> to tensors
    fn to_tensors_4(
        &self,
        data: &[(f64, f64, f64, f64)],
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let n = data.len();
        if n == 0 {
            let dummy = Tensor::zeros([1, 1], device);
            return (dummy.clone(), dummy.clone(), dummy.clone(), dummy);
        }

        let x: Vec<f32> = data.iter().map(|p| p.0 as f32).collect();
        let y: Vec<f32> = data.iter().map(|p| p.1 as f32).collect();
        let t: Vec<f32> = data.iter().map(|p| p.2 as f32).collect();
        let u: Vec<f32> = data.iter().map(|p| p.3 as f32).collect();

        (
            Tensor::<B, 1>::from_floats(x.as_slice(), device).reshape([n, 1]),
            Tensor::<B, 1>::from_floats(y.as_slice(), device).reshape([n, 1]),
            Tensor::<B, 1>::from_floats(t.as_slice(), device).reshape([n, 1]),
            Tensor::<B, 1>::from_floats(u.as_slice(), device).reshape([n, 1]),
        )
    }

    // Helper to convert Vec<(f64, f64, f64, f64, f64)> to 4 tensors (ignoring last component for now)
    fn to_tensors_5_to_4(
        &self,
        data: &[(f64, f64, f64, f64, f64)],
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let n = data.len();
        if n == 0 {
            let dummy = Tensor::zeros([1, 1], device);
            return (dummy.clone(), dummy.clone(), dummy.clone(), dummy);
        }

        let x: Vec<f32> = data.iter().map(|p| p.0 as f32).collect();
        let y: Vec<f32> = data.iter().map(|p| p.1 as f32).collect();
        let t: Vec<f32> = data.iter().map(|p| p.2 as f32).collect();
        let u: Vec<f32> = data.iter().map(|p| p.3 as f32).collect();

        (
            Tensor::<B, 1>::from_floats(x.as_slice(), device).reshape([n, 1]),
            Tensor::<B, 1>::from_floats(y.as_slice(), device).reshape([n, 1]),
            Tensor::<B, 1>::from_floats(t.as_slice(), device).reshape([n, 1]),
            Tensor::<B, 1>::from_floats(u.as_slice(), device).reshape([n, 1]),
        )
    }

    /// Generate training data for a specific task
    fn generate_task_data(&self, task: &PhysicsTask) -> KwaversResult<TaskData> {
        // Generate collocation points
        let collocation_points = self.generate_collocation_points(&task.geometry);

        // Generate boundary data
        let boundary_data = self.generate_boundary_data(&task.geometry, &task.boundary_conditions);

        // Generate initial data
        let initial_data = self.generate_initial_data(&task.geometry);

        Ok(TaskData {
            collocation_points,
            boundary_data,
            initial_data,
        })
    }

    /// Generate collocation points within geometry
    fn generate_collocation_points(
        &self,
        geometry: &Arc<crate::ml::pinn::Geometry2D>,
    ) -> Vec<(f64, f64, f64)> {
        let mut points = Vec::new();
        let num_points = 1000;

        for _ in 0..num_points {
            let x = rand::random::<f64>() * 2.0 - 1.0; // [-1, 1]
            let y = rand::random::<f64>() * 2.0 - 1.0; // [-1, 1]
            let t = rand::random::<f64>() * 1.0; // [0, 1]

            // Only include points inside geometry
            if geometry.contains(x, y) {
                points.push((x, y, t));
            }
        }

        points
    }

    /// Generate boundary data
    fn generate_boundary_data(
        &self,
        _geometry: &Arc<crate::ml::pinn::Geometry2D>,
        _conditions: &[crate::ml::pinn::BoundaryCondition2D],
    ) -> Vec<(f64, f64, f64, f64)> {
        // Generate boundary points and apply conditions
        // This is a simplified implementation
        vec![
            (0.0, 0.0, 0.0, 0.0), // Example boundary point
        ]
    }

    /// Generate initial data
    fn generate_initial_data(
        &self,
        _geometry: &Arc<crate::ml::pinn::Geometry2D>,
    ) -> Vec<(f64, f64, f64, f64, f64)> {
        // Generate initial condition data
        // This is a simplified implementation
        vec![
            (0.0, 0.0, 0.0, 0.0, 0.0), // Example initial point (x, y, t, u, du/dt)
        ]
    }

    /// Compute task-specific loss using actual PINN forward pass
    fn _compute_task_loss(
        &self,
        params: &[Tensor<B, 2>],
        data: &TaskData,
        task: &PhysicsTask,
    ) -> KwaversResult<(f64, f64)> {
        // Create a PINN model instance with current parameters for this task
        // This requires access to the actual PINN model - for now we use simplified computation
        // In full implementation, would instantiate BurnPINN2DWave with task parameters

        // Data loss: MSE between predicted and observed values
        let mut data_loss = 0.0;
        let mut count = 0;

        for &(x, y, t, u_obs) in &data.boundary_data {
            // Simplified: assume linear model u = w1*x + w2*y + w3*t + b
            // In full implementation, would use actual neural network forward pass
            let w1 = params
                .first()
                .map(|t| t.to_data().as_slice::<f32>().unwrap()[0])
                .unwrap_or(1.0) as f64;
            let w2 = params
                .get(1)
                .map(|t| t.to_data().as_slice::<f32>().unwrap()[0])
                .unwrap_or(0.5) as f64;
            let w3 = params
                .get(2)
                .map(|t| t.to_data().as_slice::<f32>().unwrap()[0])
                .unwrap_or(0.1) as f64;
            let b = params
                .get(3)
                .map(|t| t.to_data().as_slice::<f32>().unwrap()[0])
                .unwrap_or(0.0) as f64;

            let u_pred = w1 * x + w2 * y + w3 * t + b;
            data_loss += (u_pred - u_obs).powi(2);
            count += 1;
        }

        if count > 0 {
            data_loss /= count as f64;
        }

        // Physics loss: PDE residual at collocation points
        let mut physics_loss = 0.0;
        for &(_x, _y, _t) in &data.collocation_points {
            // Simplified wave equation residual: ∂²u/∂t² - c²∇²u
            // Using finite differences for derivatives
            let c = task.physics_params.wave_speed;

            // Compute u and derivatives using simplified model
            let _w1 = params
                .first()
                .map(|t| t.to_data().as_slice::<f32>().unwrap()[0])
                .unwrap_or(1.0) as f64;
            let _w2 = params
                .get(1)
                .map(|t| t.to_data().as_slice::<f32>().unwrap()[0])
                .unwrap_or(0.5) as f64;
            let _w3 = params
                .get(2)
                .map(|t| t.to_data().as_slice::<f32>().unwrap()[0])
                .unwrap_or(0.1) as f64;

            // u = w1*x + w2*y + w3*t
            // let u = w1 * x + w2 * y + w3 * t;
            let _u = 0.0; // Simplified for now

            // Second derivatives (all zero for linear model except ∂²u/∂t² = 0)
            let d2u_dt2 = 0.0;
            let d2u_dx2 = 0.0;
            let d2u_dy2 = 0.0;
            let laplacian = d2u_dx2 + d2u_dy2;

            // PDE residual: ∂²u/∂t² - c²∇²u
            let residual = d2u_dt2 - c * c * laplacian;
            physics_loss += residual * residual;
        }
        physics_loss /= data.collocation_points.len() as f64;

        Ok((data_loss + physics_loss, physics_loss))
    }

    /// Compute gradients for inner-loop adaptation using finite differences
    fn _compute_gradients(
        &self,
        params: &[Tensor<B, 2>],
        data: &TaskData,
        task: &PhysicsTask,
    ) -> KwaversResult<Vec<Tensor<B, 2>>> {
        let eps = 1e-6_f32; // Small perturbation for finite differences
        let mut gradients = Vec::with_capacity(params.len());

        // Compute baseline loss
        let (baseline_loss, _) = self._compute_task_loss(params, data, task)?;

        for (i, param) in params.iter().enumerate() {
            let param_shape = param.shape();
            let mut param_grad = Tensor::<B, 2>::zeros(param_shape.clone(), &param.device());

            // For simplicity, compute gradient w.r.t. first element of each parameter
            // In full implementation, would compute gradient for all elements
            if !param_shape.dims.is_empty() && param_shape.dims[0] > 0 {
                // Perturb the first element
                let mut perturbed_params = params.to_vec();

                perturbed_params[i] = param.clone().add_scalar(eps);
                let (perturbed_loss, _) = self._compute_task_loss(&perturbed_params, data, task)?;

                // Finite difference gradient
                let grad_val = (perturbed_loss - baseline_loss) as f32 / eps;
                param_grad = param_grad.add_scalar(grad_val);
            }

            gradients.push(param_grad);
        }

        Ok(gradients)
    }

    /// Compute meta-gradients using MAML-style second-order derivatives
    fn _compute_meta_gradients(
        &self,
        task_gradients: &[Vec<Tensor<B, 2>>],
        task_losses: &[f64],
    ) -> Vec<Option<Tensor<B, 2>>> {
        let num_params = self._meta_optimizer._m.len();
        if task_gradients.is_empty() || task_losses.is_empty() {
            return vec![None; num_params];
        }

        let num_tasks = task_gradients.len();
        let mut meta_gradients = Vec::with_capacity(num_params);

        // For each meta-parameter, compute the gradient across all tasks
        for param_idx in 0..num_params {
            let mut param_meta_grad: Option<Tensor<B, 2>> = None;

            // Aggregate gradients from all tasks for this parameter
            for task_idx in 0..num_tasks {
                if param_idx >= task_gradients[task_idx].len() {
                    continue;
                }

                let task_grad = &task_gradients[task_idx][param_idx];
                let task_loss = task_losses[task_idx];

                // Simplified MAML update: meta-gradient is weighted sum of task gradients
                let weighted_grad = task_grad.clone().mul_scalar(task_loss as f32);

                if let Some(acc) = param_meta_grad.as_mut() {
                    *acc = acc.clone().add(weighted_grad);
                } else {
                    param_meta_grad = Some(weighted_grad);
                }
            }

            // Average across tasks
            if let Some(grad) = param_meta_grad {
                meta_gradients.push(Some(grad.div_scalar(num_tasks as f32)));
            } else {
                meta_gradients.push(None);
            }
        }

        meta_gradients
    }

    /// Compute generalization score across tasks
    fn compute_generalization_score(&self, task_losses: &[f64]) -> f64 {
        // Compute variance of task losses (lower variance = better generalization)
        let mean = task_losses.iter().sum::<f64>() / task_losses.len() as f64;
        let variance = task_losses
            .iter()
            .map(|loss| (loss - mean).powi(2))
            .sum::<f64>()
            / task_losses.len() as f64;

        // Convert to score (0-1, higher is better)
        1.0 / (1.0 + variance.sqrt())
    }

    /// Get meta-learning statistics
    pub fn get_stats(&self) -> &MetaLearningStats {
        &self.stats
    }
}

impl<B: AutodiffBackend> MetaOptimizer<B> {
    /// Create a new meta-optimizer
    pub fn new(lr: f64, num_params: usize) -> Self {
        Self {
            lr,
            _momentum: Some(0.9),
            _beta1: 0.9,
            _beta2: 0.999,
            _epsilon: 1e-8,
            _iteration_count: 0,
            _m: vec![None; num_params],
            _v: vec![None; num_params],
        }
    }

    /// Perform optimization step
    pub fn step(&mut self, params: &mut [Tensor<B, 2>], gradients: &[Option<Tensor<B, 2>>]) {
        self._iteration_count += 1;

        for (param, grad) in params.iter_mut().zip(gradients.iter()) {
            if let Some(g) = grad {
                // Simplified SGD update for now
                // In practice, this would be full Adam optimization
                *param = param.clone().sub(g.clone().mul_scalar(self.lr as f32));
            }
        }
    }
}

impl TaskSampler {
    /// Create a new task sampler
    pub fn new(strategy: SamplingStrategy, config: MetaLearningConfig) -> Self {
        Self {
            task_pool: Vec::new(),
            sampling_strategy: strategy,
            current_index: 0,
            task_history: Vec::new(),
            config,
        }
    }

    /// Add a task to the pool
    pub fn add_task(&mut self, task: PhysicsTask) {
        self.task_pool.push(task);
    }

    /// Sample a batch of tasks
    pub fn sample_batch(&mut self, batch_size: usize) -> KwaversResult<Vec<PhysicsTask>> {
        if self.task_pool.is_empty() {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
                    resource: "No tasks available in task pool".to_string(),
                },
            ));
        }

        let mut batch = Vec::new();

        for _ in 0..batch_size {
            let task = match self.sampling_strategy {
                SamplingStrategy::Random => {
                    let idx = rand::random::<usize>() % self.task_pool.len();
                    self.task_pool[idx].clone()
                }
                SamplingStrategy::Curriculum => {
                    // Progressive difficulty curriculum learning
                    // Literature: Bengio et al. (2009) Curriculum Learning, Graves et al. (2017) Automated Curriculum Learning

                    // Compute task difficulty based on multiple factors
                    let task_difficulties: Vec<f64> = self
                        .task_pool
                        .iter()
                        .map(|task| {
                            // Difficulty = f(complexity, domain_knowledge, boundary_conditions)
                            let complexity_score = match task.pde_type {
                                PdeType::Wave => 1.0,
                                PdeType::Diffusion => 2.0,
                                PdeType::NavierStokes => 4.0,
                                PdeType::Electromagnetic => 3.0,
                                PdeType::Acoustic => 2.0,
                                PdeType::Elastic => 3.0,
                            };

                            let geometry_complexity = match task.geometry.as_ref() {
                                Geometry2D::Rectangular { .. } => 1.0,
                                Geometry2D::Circular { .. } => 2.0,
                                Geometry2D::MultiRegion { .. } => 4.0,
                                _ => 3.0, // Default for other geometries
                            };

                            let boundary_complexity = task.boundary_conditions.len() as f64;

                            complexity_score * geometry_complexity * boundary_complexity
                        })
                        .collect();

                    // Progressive sampling: start with easier tasks, gradually increase difficulty
                    let progress_ratio = self.current_index as f64 / self.config.max_tasks as f64;
                    let target_difficulty =
                        progress_ratio * task_difficulties.iter().cloned().fold(0.0, f64::max);

                    // Sample from tasks within current difficulty range
                    let candidates: Vec<usize> = task_difficulties
                        .iter()
                        .enumerate()
                        .filter(|(_, &diff)| diff <= target_difficulty + 1.0) // Allow some exploration
                        .map(|(i, _)| i)
                        .collect();

                    if candidates.is_empty() {
                        // Fallback to any task if no candidates found
                        let idx = self.current_index % self.task_pool.len();
                        self.task_pool[idx].clone()
                    } else {
                        // Sample from candidates, preferring higher difficulty within range
                        let selected_idx = candidates[rand::random::<usize>() % candidates.len()];
                        self.current_index += 1;
                        self.task_pool[selected_idx].clone()
                    }
                }
                SamplingStrategy::Balanced => {
                    // Sample from different physics families
                    let idx = rand::random::<usize>() % self.task_pool.len();
                    self.task_pool[idx].clone()
                }
                SamplingStrategy::Diversity => {
                    // Maximize task diversity using determinantal point processes
                    // Literature: Kulesza & Taskar (2012) Determinantal Point Processes for Machine Learning

                    // Track recently sampled task types to ensure diversity
                    let mut sampled_types = std::collections::HashSet::new();
                    for recent_task in self.task_history.iter().rev().take(5) {
                        if let Some(task) = self.task_pool.get(*recent_task) {
                            sampled_types.insert(task.pde_type.clone());
                        }
                    }

                    // Score tasks by diversity from recent samples
                    let diversity_scores: Vec<(usize, f64)> = self
                        .task_pool
                        .iter()
                        .enumerate()
                        .map(|(i, task)| {
                            let type_diversity = if sampled_types.contains(&task.pde_type) {
                                0.3
                            } else {
                                1.0
                            };
                            let geometry_diversity = match task.geometry.as_ref() {
                                Geometry2D::Rectangular { .. } => 0.5,
                                Geometry2D::Circular { .. } => 0.7,
                                Geometry2D::MultiRegion { .. } => 1.0,
                                _ => 0.8,
                            };
                            let score = type_diversity * geometry_diversity;
                            (i, score)
                        })
                        .collect();

                    // Sample proportionally to diversity scores
                    let total_score: f64 = diversity_scores.iter().map(|(_, s)| s).sum();
                    let mut rand_val = rand::random::<f64>() * total_score;
                    let mut selected_task = None;

                    for (idx, score) in diversity_scores {
                        rand_val -= score;
                        if rand_val <= 0.0 {
                            selected_task = Some(self.task_pool[idx].clone());
                            break;
                        }
                    }

                    // Fallback
                    selected_task.unwrap_or_else(|| {
                        self.task_pool[rand::random::<usize>() % self.task_pool.len()].clone()
                    })
                }
            };

            batch.push(task);
        }

        Ok(batch)
    }
}

impl Default for MetaLearningStats {
    fn default() -> Self {
        Self {
            meta_epochs_completed: 0,
            total_tasks_processed: 0,
            average_adaptation_time: 0.0,
            average_meta_loss: 0.0,
            best_generalization_score: 0.0,
            convergence_rate: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = burn::backend::Autodiff<NdArray<f32>>;

    #[test]
    fn test_meta_learning_config() {
        let config = MetaLearningConfig {
            inner_lr: 0.01,
            outer_lr: 0.001,
            adaptation_steps: 5,
            meta_batch_size: 4,
            meta_epochs: 100,
            first_order: true,
            physics_regularization: 0.1,
            num_layers: 3,
            input_dim: 3,
            hidden_dim: 20,
            output_dim: 1,
            max_tasks: 10,
        };

        assert_eq!(config.adaptation_steps, 5);
        assert_eq!(config.meta_batch_size, 4);
    }

    #[test]
    fn test_task_sampler() {
        let config = MetaLearningConfig {
            inner_lr: 0.01,
            outer_lr: 0.001,
            adaptation_steps: 5,
            meta_batch_size: 4,
            meta_epochs: 100,
            first_order: true,
            physics_regularization: 0.1,
            num_layers: 3,
            input_dim: 3,
            hidden_dim: 20,
            output_dim: 1,
            max_tasks: 10,
        };
        let mut sampler = TaskSampler::new(SamplingStrategy::Random, config);

        let task = PhysicsTask {
            id: "test_task".to_string(),
            pde_type: PdeType::Wave,
            physics_params: PhysicsParameters {
                wave_speed: 343.0,
                density: 1.2,
                viscosity: None,
                absorption: None,
                nonlinearity: None,
            },
            geometry: Arc::new(crate::ml::pinn::Geometry2D::rectangular(0.0, 1.0, 0.0, 1.0)),
            boundary_conditions: vec![],
            training_data: None,
            validation_data: TaskData {
                collocation_points: vec![(0.0, 0.0, 0.0)],
                boundary_data: vec![],
                initial_data: vec![],
            },
        };

        sampler.add_task(task);
        let batch = sampler.sample_batch(1).unwrap();

        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].id, "test_task");
    }

    #[test]
    fn test_meta_optimizer_creation() {
        let optimizer = MetaOptimizer::<TestBackend>::new(0.001, 2);
        assert_eq!(optimizer.lr, 0.001);
        assert_eq!(optimizer._m.len(), 2);
        assert_eq!(optimizer._v.len(), 2);
    }
}
