//! Meta-Learner Core Implementation
//!
//! Implements the MAML (Model-Agnostic Meta-Learning) algorithm for Physics-Informed
//! Neural Networks, enabling fast adaptation to new physics problems.

use crate::solver::inverse::pinn::ml::burn_wave_equation_2d::{
    BurnLossWeights2D, BurnPINN2DConfig, BurnPINN2DWave, SimpleOptimizer2D,
};
use crate::solver::inverse::pinn::ml::meta_learning::config::MetaLearningConfig;
use crate::solver::inverse::pinn::ml::meta_learning::gradient::{GradientApplicator, GradientExtractor};
use crate::solver::inverse::pinn::ml::meta_learning::metrics::{MetaLearningStats, MetaLoss};
use crate::solver::inverse::pinn::ml::meta_learning::optimizer::MetaOptimizer;
use crate::solver::inverse::pinn::ml::meta_learning::sampling::SamplingStrategy;
use crate::solver::inverse::pinn::ml::meta_learning::sampling::TaskSampler;
use crate::solver::inverse::pinn::ml::meta_learning::types::{PhysicsTask, TaskData};
use crate::core::error::KwaversResult;
use burn::module::Module;
use burn::prelude::ToElement;
use burn::tensor::{backend::AutodiffBackend, Tensor};
use std::sync::Arc;

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
            let mut extractor = GradientExtractor::new(task_grad_set);
            // Map over model structure to extract gradients in order
            self.base_model.clone().map(&mut extractor);
            let task_grads_flat = extractor.into_gradients();

            if i == 0 {
                accumulated_grads = task_grads_flat;
            } else {
                for (acc, new) in accumulated_grads.iter_mut().zip(task_grads_flat.iter()) {
                    if let (Some(a), Some(b)) = (acc.as_mut(), new.as_ref()) {
                        *a = a.clone() + b.clone();
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
        let mut applicator = GradientApplicator::new(averaged_grads, self.config.outer_lr);

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
        geometry: &Arc<crate::solver::inverse::pinn::ml::Geometry2D>,
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
        _geometry: &Arc<crate::solver::inverse::pinn::ml::Geometry2D>,
        _conditions: &[crate::solver::inverse::pinn::ml::BoundaryCondition2D],
    ) -> Vec<(f64, f64, f64, f64)> {
        // TODO_AUDIT: P1 - Meta-Learning Boundary Data Generation - Simplified Stub
        //
        // PROBLEM:
        // Returns a single dummy boundary point (0,0,0,0) regardless of geometry or boundary conditions.
        // Meta-learner cannot adapt to tasks with specific boundary constraints.
        //
        // IMPACT:
        // - Meta-learned model initialization ignores boundary condition structure
        // - Transfer learning to new tasks with different BCs requires full retraining
        // - Defeats purpose of meta-learning for boundary-dominated problems
        // - Blocks applications: Dirichlet/Neumann/Robin BC adaptation, complex geometry handling
        // - Severity: P1 (advanced research feature)
        //
        // REQUIRED IMPLEMENTATION:
        // 1. Parse geometry to extract boundary curves/surfaces
        // 2. Sample points uniformly along boundary (e.g., 100-500 points depending on complexity)
        // 3. For each boundary condition type:
        //    - Dirichlet: (x, y, t, u_bc) where u_bc is prescribed value
        //    - Neumann: (x, y, t, ∂u/∂n) where ∂u/∂n is prescribed normal derivative
        //    - Robin: (x, y, t, αu + β∂u/∂n) mixed condition
        // 4. Return Vec<(x, y, t, bc_value)> with proper spatial/temporal sampling
        //
        // MATHEMATICAL SPECIFICATION:
        // Boundary sampling strategy:
        //   - Spatial: Arc-length parameterization s ∈ [0, L] with uniform Δs
        //   - Temporal: t ∈ [0, T_max] with uniform Δt (if time-dependent BCs)
        //   - Target: O(√N) boundary points for N interior collocation points
        //
        // VALIDATION CRITERIA:
        // - Test: Rectangular domain [0,1]×[0,1] with Dirichlet u=0 on all sides
        //   Should return ~100 boundary points with bc_value=0
        // - Test: Circle with Neumann ∂u/∂r = 1
        //   Should return points on circumference with bc_value=1
        // - Coverage: Boundary point spacing < 2× interior collocation spacing
        //
        // REFERENCES:
        // - Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation" (ICML 2017)
        // - Raissi et al., "Physics-informed neural networks" (boundary sampling strategies)
        //
        // ESTIMATED EFFORT: 8-12 hours
        // - Implementation: 6-8 hours (geometry parsing, sampling, BC application)
        // - Testing: 2-3 hours (geometric primitives, BC types)
        // - Documentation: 1 hour
        //
        // ASSIGNED: Sprint 212 (Meta-Learning Enhancement)
        // PRIORITY: P1 (Research feature - meta-learning BC adaptation)

        // Generate boundary points and apply conditions
        // This is a simplified implementation
        vec![
            (0.0, 0.0, 0.0, 0.0), // Example boundary point
        ]
    }

    /// Generate initial data
    fn generate_initial_data(
        &self,
        _geometry: &Arc<crate::solver::inverse::pinn::ml::Geometry2D>,
    ) -> Vec<(f64, f64, f64, f64, f64)> {
        // TODO_AUDIT: P1 - Meta-Learning Initial Condition Data Generation - Simplified Stub
        //
        // PROBLEM:
        // Returns a single dummy initial condition point (0,0,0,0,0) regardless of geometry.
        // Meta-learner cannot adapt to tasks with specific initial conditions.
        //
        // IMPACT:
        // - Meta-learned model initialization ignores IC structure (Gaussian pulse, plane wave, etc.)
        // - Transfer learning to new ICs requires full retraining from scratch
        // - Reduces meta-learning effectiveness for time-dependent problems
        // - Blocks applications: IC-sensitive dynamics (wave packets, thermal pulses, shock initialization)
        // - Severity: P1 (advanced research feature)
        //
        // REQUIRED IMPLEMENTATION:
        // 1. Sample spatial points (x, y) uniformly or quasi-randomly within geometry
        // 2. Set t = 0 for all initial condition points
        // 3. Compute initial values u(x,y,0) from specified IC function:
        //    - For wave equation: u₀(x,y) and ∂u/∂t|_{t=0} = v₀(x,y)
        //    - For diffusion: u₀(x,y) only
        //    - For general hyperbolic: u₀(x,y) and higher time derivatives if needed
        // 4. Return Vec<(x, y, t=0, u₀, v₀)> with sufficient spatial coverage
        //
        // MATHEMATICAL SPECIFICATION:
        // Initial condition sampling:
        //   - Spatial coverage: N_ic = O(√N) where N is total collocation points
        //   - Sampling methods: Uniform grid, Sobol sequence, or Latin hypercube
        //   - For wave equation: Both u(x,y,0) and ∂u/∂t(x,y,0) required
        //
        // Common IC patterns to support:
        //   - Gaussian pulse: u₀(x,y) = A exp(-((x-x₀)² + (y-y₀)²)/(2σ²))
        //   - Plane wave: u₀(x,y) = A sin(k_x·x + k_y·y)
        //   - Dirac delta (smoothed): u₀(x,y) = δ_ε(x-x₀, y-y₀)
        //
        // VALIDATION CRITERIA:
        // - Test: 2D domain [0,1]×[0,1] with Gaussian IC centered at (0.5, 0.5)
        //   Should return ~100 points with u₀ = exp(-r²/2σ²), v₀ = 0
        // - Test: Plane wave IC with k = (2π, 0)
        //   Should return u₀ = sin(2πx), v₀ computed from dispersion relation
        // - Coverage: At least 50-200 IC points for typical 2D problems
        //
        // REFERENCES:
        // - Raissi et al., "Physics-informed neural networks" (initial condition handling)
        // - Finn et al., "Model-Agnostic Meta-Learning" (MAML algorithm)
        //
        // ESTIMATED EFFORT: 6-10 hours
        // - Implementation: 4-6 hours (spatial sampling, IC function evaluation)
        // - Testing: 2-3 hours (Gaussian, plane wave, custom ICs)
        // - Documentation: 1 hour
        //
        // ASSIGNED: Sprint 212 (Meta-Learning Enhancement)
        // PRIORITY: P1 (Research feature - meta-learning IC adaptation)

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
        let num_params = self.base_model.parameters().len();
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
