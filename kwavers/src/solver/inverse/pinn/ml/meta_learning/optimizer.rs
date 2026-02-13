//! Meta-Optimizer for Outer-Loop Updates
//!
//! This module implements the meta-optimizer responsible for updating meta-parameters
//! based on aggregated gradients from multiple task adaptations.
//!
//! # Outer-Loop Optimization
//!
//! In MAML, the meta-optimizer updates the initial model parameters (meta-parameters)
//! to optimize for fast adaptation across the task distribution:
//!
//! ```text
//! θ ← θ - β∇_θ Σ_τ L_τ(U_τ(θ))
//! ```
//!
//! where:
//! - θ: meta-parameters
//! - β: outer-loop learning rate
//! - τ: task from distribution
//! - U_τ(θ): adapted parameters after inner-loop updates
//! - L_τ: task-specific validation loss
//!
//! # Optimization Algorithms
//!
//! Current implementation supports:
//! - **SGD**: Simple gradient descent
//! - **Momentum**: SGD with first-moment accumulation
//! - **Adam**: Adaptive learning rates with moment estimation
//! - **RMSProp**: Adaptive learning rates with running second moment
//!
//! # Literature References
//!
//! 1. Finn, C., Abbeel, P., & Levine, S. (2017).
//!    "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
//!    *ICML 2017*
//!
//! 2. Kingma, D. P., & Ba, J. (2014).
//!    "Adam: A Method for Stochastic Optimization"
//!    *arXiv:1412.6980*
//!
//! 3. Antoniou, A., Edwards, H., & Storkey, A. (2018).
//!    "How to train your MAML"
//!    *ICLR 2019*
//!    - Recommends Adam optimizer for meta-learning
//!
//! # Examples
//!
//! ```rust,ignore
//! use kwavers::solver::inverse::pinn::ml::meta_learning::MetaOptimizer;
//!
//! // Create meta-optimizer for 100 parameters
//! let mut optimizer = MetaOptimizer::new(0.001, 100);
//!
//! // Update parameters with aggregated meta-gradients
//! optimizer.step(&mut params, &meta_gradients);
//! ```

use burn::tensor::{backend::AutodiffBackend, Tensor};

/// Meta-optimizer for outer-loop parameter updates
///
/// Implements optimization algorithms for updating meta-parameters based on
/// aggregated gradients from multiple task adaptations.
///
/// # Optimization Strategy
///
/// The meta-optimizer uses configurable SGD, momentum, Adam, or RMSProp updates:
/// TODO_AUDIT: P2 - Advanced Meta-Optimizers - Implement second-order methods for meta-training
/// DEPENDS ON: solver/inverse/pinn/ml/optimizer/adam_meta.rs, math/optimization/quasi_newton.rs
/// MISSING: Adaptive moment estimation for meta-parameters
/// MISSING: BFGS quasi-Newton method for meta-optimization
/// MISSING: Trust-region methods for meta-learning stability
/// MISSING: Line search algorithms for optimal step sizes
/// ```text
/// θ ← θ - α∇L_meta
/// ```
///
/// For improved convergence, momentum can be enabled (future work):
/// ```text
/// v ← β*v + ∇L_meta
/// θ ← θ - α*v
/// ```
///
/// # Adam Support
///
/// Adam optimizer with adaptive learning rates:
/// ```text
/// m ← β₁*m + (1-β₁)*∇L
/// v ← β₂*v + (1-β₂)*∇L²
/// m̂ ← m/(1-β₁ᵗ)
/// v̂ ← v/(1-β₂ᵗ)
/// θ ← θ - α*m̂/(√v̂ + ε)
/// ```
#[derive(Debug)]
pub struct MetaOptimizer<B: AutodiffBackend> {
    /// Outer-loop learning rate
    lr: f64,

    /// Optimizer mode
    mode: MetaOptimizerMode,

    /// Momentum parameter (0.0 = no momentum, 0.9 = typical)
    _momentum: Option<f64>,

    /// Adam beta1 parameter (first moment decay)
    _beta1: f64,

    /// Adam beta2 parameter (second moment decay)
    _beta2: f64,

    /// Adam epsilon for numerical stability
    _epsilon: f64,

    /// Iteration count for bias correction
    _iteration_count: usize,

    /// First moment estimates (for Adam/momentum)
    _m: Vec<Option<Tensor<B, 2>>>,

    /// Second moment estimates (for Adam/RMSProp)
    _v: Vec<Option<Tensor<B, 2>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaOptimizerMode {
    Sgd,
    Momentum,
    Adam,
    RmsProp,
}

impl<B: AutodiffBackend> MetaOptimizer<B> {
    /// Create a new meta-optimizer
    ///
    /// # Arguments
    /// - `lr`: Outer-loop learning rate (typical: 0.0001 - 0.01)
    /// - `num_params`: Number of parameter tensors to optimize
    ///
    /// # Panics
    /// Panics if learning rate is non-positive or NaN
    pub fn new(lr: f64, num_params: usize) -> Self {
        assert!(
            lr > 0.0 && lr.is_finite(),
            "Learning rate must be positive and finite, got {}",
            lr
        );

        Self {
            lr,
            mode: MetaOptimizerMode::Sgd,
            _momentum: None,
            _beta1: 0.9,
            _beta2: 0.999,
            _epsilon: 1e-8,
            _iteration_count: 0,
            _m: vec![None; num_params],
            _v: vec![None; num_params],
        }
    }

    /// Create meta-optimizer with custom momentum
    pub fn with_momentum(lr: f64, num_params: usize, momentum: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&momentum),
            "Momentum must be in [0, 1), got {}",
            momentum
        );

        let mut optimizer = Self::new(lr, num_params);
        optimizer._momentum = Some(momentum);
        optimizer.mode = MetaOptimizerMode::Momentum;
        optimizer
    }

    /// Create meta-optimizer with Adam hyperparameters
    pub fn with_adam(lr: f64, num_params: usize, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        assert!(
            beta1 > 0.0 && beta1 < 1.0,
            "Beta1 must be in (0, 1), got {}",
            beta1
        );
        assert!(
            beta2 > 0.0 && beta2 < 1.0,
            "Beta2 must be in (0, 1), got {}",
            beta2
        );
        assert!(epsilon > 0.0, "Epsilon must be positive, got {}", epsilon);

        let mut optimizer = Self::new(lr, num_params);
        optimizer._beta1 = beta1;
        optimizer._beta2 = beta2;
        optimizer._epsilon = epsilon;
        optimizer.mode = MetaOptimizerMode::Adam;
        optimizer
    }

    /// Create meta-optimizer with RMSProp hyperparameters
    pub fn with_rmsprop(lr: f64, num_params: usize, decay: f64, epsilon: f64) -> Self {
        assert!(
            decay > 0.0 && decay < 1.0,
            "Decay must be in (0, 1), got {}",
            decay
        );
        assert!(epsilon > 0.0, "Epsilon must be positive, got {}", epsilon);

        let mut optimizer = Self::new(lr, num_params);
        optimizer._beta2 = decay;
        optimizer._epsilon = epsilon;
        optimizer.mode = MetaOptimizerMode::RmsProp;
        optimizer
    }

    /// Perform optimization step (simplified SGD)
    ///
    /// Updates parameters using vanilla gradient descent:
    /// ```text
    /// θ ← θ - α∇L
    /// ```
    ///
    /// # Arguments
    /// - `params`: Mutable reference to model parameters
    /// - `gradients`: Meta-gradients from task distribution
    ///
    /// # Notes
    /// - Gradients should be averaged across tasks before calling this method
    /// - None gradients leave corresponding parameters unchanged
    /// - Momentum and Adam optimizers are planned for future implementation
    pub fn step(&mut self, params: &mut [Tensor<B, 2>], gradients: &[Option<Tensor<B, 2>>]) {
        self._iteration_count += 1;

        if self._m.len() != params.len() {
            self._m = vec![None; params.len()];
        }
        if self._v.len() != params.len() {
            self._v = vec![None; params.len()];
        }

        let beta1 = self._beta1 as f32;
        let beta2 = self._beta2 as f32;
        let eps = self._epsilon as f32;
        let beta1_pow = (1.0 - self._beta1.powi(self._iteration_count as i32)) as f32;
        let beta2_pow = (1.0 - self._beta2.powi(self._iteration_count as i32)) as f32;

        for (idx, (param, grad)) in params.iter_mut().zip(gradients.iter()).enumerate() {
            if let Some(g) = grad {
                match self.mode {
                    MetaOptimizerMode::Sgd => {
                        *param = param.clone().sub(g.clone().mul_scalar(self.lr as f32));
                    }
                    MetaOptimizerMode::Momentum => {
                        let momentum = self._momentum.unwrap_or(0.0) as f32;
                        let prev = self._m[idx].take();
                        let m = if let Some(m_prev) = prev {
                            m_prev.mul_scalar(momentum).add(g.clone())
                        } else {
                            g.clone()
                        };
                        *param = param.clone().sub(m.clone().mul_scalar(self.lr as f32));
                        self._m[idx] = Some(m);
                    }
                    MetaOptimizerMode::Adam => {
                        let prev_m = self._m[idx].take();
                        let prev_v = self._v[idx].take();

                        let m = if let Some(m_prev) = prev_m {
                            m_prev.mul_scalar(beta1).add(g.clone().mul_scalar(1.0 - beta1))
                        } else {
                            g.clone().mul_scalar(1.0 - beta1)
                        };

                        let g2 = g.clone().powf_scalar(2.0);
                        let v = if let Some(v_prev) = prev_v {
                            v_prev.mul_scalar(beta2).add(g2.mul_scalar(1.0 - beta2))
                        } else {
                            g2.mul_scalar(1.0 - beta2)
                        };

                        let m_hat = m.clone().div_scalar(beta1_pow);
                        let v_hat = v.clone().div_scalar(beta2_pow);
                        let denom = v_hat.sqrt().add_scalar(eps);

                        *param = param
                            .clone()
                            .sub((m_hat / denom).mul_scalar(self.lr as f32));

                        self._m[idx] = Some(m);
                        self._v[idx] = Some(v);
                    }
                    MetaOptimizerMode::RmsProp => {
                        let prev_v = self._v[idx].take();
                        let g2 = g.clone().powf_scalar(2.0);
                        let v = if let Some(v_prev) = prev_v {
                            v_prev.mul_scalar(beta2).add(g2.mul_scalar(1.0 - beta2))
                        } else {
                            g2.mul_scalar(1.0 - beta2)
                        };

                        let denom = v.sqrt().add_scalar(eps);
                        *param = param
                            .clone()
                            .sub((g.clone() / denom).mul_scalar(self.lr as f32));

                        self._v[idx] = Some(v);
                    }
                }
            }
        }
    }

    /// Get current learning rate
    pub fn learning_rate(&self) -> f64 {
        self.lr
    }

    /// Set learning rate (for learning rate scheduling)
    pub fn set_learning_rate(&mut self, lr: f64) {
        assert!(
            lr > 0.0 && lr.is_finite(),
            "Learning rate must be positive and finite, got {}",
            lr
        );
        self.lr = lr;
    }

    /// Get number of optimization steps performed
    pub fn iteration_count(&self) -> usize {
        self._iteration_count
    }

    /// Reset optimizer state (clears momentum buffers)
    pub fn reset(&mut self) {
        self._iteration_count = 0;
        self._m.iter_mut().for_each(|m| *m = None);
        self._v.iter_mut().for_each(|v| *v = None);
    }

    /// Apply learning rate decay
    ///
    /// Common decay schedules:
    /// - Step decay: multiply by factor every N epochs
    /// - Exponential: lr = lr₀ * e^(-λt)
    /// - Cosine annealing: lr = lr_min + 0.5(lr_max - lr_min)(1 + cos(πt/T))
    pub fn decay_learning_rate(&mut self, factor: f64) {
        assert!(
            factor > 0.0 && factor <= 1.0,
            "Decay factor must be in (0, 1], got {}",
            factor
        );
        self.lr *= factor;
    }
}

/// Learning rate schedule for meta-optimization
///
/// Implements common learning rate scheduling strategies for improved convergence.
#[derive(Debug, Clone)]
pub enum LearningRateSchedule {
    /// Constant learning rate
    Constant,

    /// Step decay: multiply by factor every N epochs
    StepDecay { factor: f64, step_size: usize },

    /// Exponential decay: lr = lr₀ * e^(-λt)
    Exponential { decay_rate: f64 },

    /// Cosine annealing: smooth decay following cosine curve
    CosineAnnealing { lr_min: f64, total_epochs: usize },
}

impl LearningRateSchedule {
    /// Compute learning rate for given epoch
    pub fn get_lr(&self, epoch: usize, lr_initial: f64) -> f64 {
        match self {
            LearningRateSchedule::Constant => lr_initial,

            LearningRateSchedule::StepDecay { factor, step_size } => {
                let num_decays = epoch / step_size;
                lr_initial * factor.powi(num_decays as i32)
            }

            LearningRateSchedule::Exponential { decay_rate } => {
                lr_initial * (-decay_rate * epoch as f64).exp()
            }

            LearningRateSchedule::CosineAnnealing {
                lr_min,
                total_epochs,
            } => {
                let progress = (epoch as f64 / *total_epochs as f64).min(1.0);
                lr_min
                    + 0.5 * (lr_initial - lr_min) * (1.0 + (std::f64::consts::PI * progress).cos())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = burn::backend::Autodiff<NdArray<f32>>;

    fn scalar_f32(tensor: &Tensor<TestBackend, 2>) -> f32 {
        let data = tensor.clone().into_data();
        let slice = data.as_slice::<f32>().expect("tensor data must be f32");
        slice[0]
    }

    #[test]
    fn test_meta_optimizer_creation() {
        let optimizer = MetaOptimizer::<TestBackend>::new(0.001, 10);
        assert_eq!(optimizer.learning_rate(), 0.001);
        assert_eq!(optimizer.iteration_count(), 0);
    }

    #[test]
    #[should_panic(expected = "Learning rate must be positive")]
    fn test_meta_optimizer_invalid_lr() {
        let _ = MetaOptimizer::<TestBackend>::new(-0.001, 10);
    }

    #[test]
    fn test_meta_optimizer_with_momentum() {
        let optimizer = MetaOptimizer::<TestBackend>::with_momentum(0.001, 10, 0.9);
        assert_eq!(optimizer.learning_rate(), 0.001);
        assert_eq!(optimizer._momentum, Some(0.9));
    }

    #[test]
    fn test_meta_optimizer_with_adam() {
        let optimizer = MetaOptimizer::<TestBackend>::with_adam(0.001, 10, 0.9, 0.999, 1e-8);
        assert_eq!(optimizer.learning_rate(), 0.001);
        assert_eq!(optimizer._beta1, 0.9);
        assert_eq!(optimizer._beta2, 0.999);
        assert_eq!(optimizer._epsilon, 1e-8);
    }

    #[test]
    fn test_set_learning_rate() {
        let mut optimizer = MetaOptimizer::<TestBackend>::new(0.001, 10);
        optimizer.set_learning_rate(0.0005);
        assert_eq!(optimizer.learning_rate(), 0.0005);
    }

    #[test]
    fn test_decay_learning_rate() {
        let mut optimizer = MetaOptimizer::<TestBackend>::new(0.001, 10);
        optimizer.decay_learning_rate(0.5);
        assert!((optimizer.learning_rate() - 0.0005).abs() < 1e-10);
    }

    #[test]
    fn test_reset_optimizer() {
        let mut optimizer = MetaOptimizer::<TestBackend>::new(0.001, 10);
        optimizer._iteration_count = 100;
        optimizer.reset();
        assert_eq!(optimizer.iteration_count(), 0);
    }

    #[test]
    fn test_lr_schedule_constant() {
        let schedule = LearningRateSchedule::Constant;
        assert_eq!(schedule.get_lr(0, 0.001), 0.001);
        assert_eq!(schedule.get_lr(100, 0.001), 0.001);
    }

    #[test]
    fn test_lr_schedule_step_decay() {
        let schedule = LearningRateSchedule::StepDecay {
            factor: 0.5,
            step_size: 10,
        };
        assert_eq!(schedule.get_lr(0, 0.001), 0.001);
        assert_eq!(schedule.get_lr(10, 0.001), 0.0005);
        assert_eq!(schedule.get_lr(20, 0.001), 0.00025);
    }

    #[test]
    fn test_lr_schedule_exponential() {
        let schedule = LearningRateSchedule::Exponential { decay_rate: 0.01 };
        let lr0 = schedule.get_lr(0, 0.001);
        let lr100 = schedule.get_lr(100, 0.001);
        assert!((lr0 - 0.001).abs() < 1e-10);
        assert!(lr100 < lr0); // Should decay
    }

    #[test]
    fn test_lr_schedule_cosine_annealing() {
        let schedule = LearningRateSchedule::CosineAnnealing {
            lr_min: 0.0001,
            total_epochs: 1000,
        };
        let lr0 = schedule.get_lr(0, 0.001);
        let lr_mid = schedule.get_lr(500, 0.001);
        let lr_end = schedule.get_lr(1000, 0.001);

        assert!((lr0 - 0.001).abs() < 1e-10);
        assert!(lr_mid > 0.0001 && lr_mid < 0.001);
        assert!((lr_end - 0.0001).abs() < 1e-6);
    }

    #[test]
    fn test_adam_step_updates_parameters() {
        let device = Default::default();
        let mut optimizer = MetaOptimizer::<TestBackend>::with_adam(0.01, 1, 0.9, 0.999, 1e-8);

        let mut params = vec![Tensor::<TestBackend, 2>::zeros([1, 1], &device)];
        let grads = vec![Some(Tensor::<TestBackend, 2>::ones([1, 1], &device))];

        optimizer.step(&mut params, &grads);
        assert!(scalar_f32(&params[0]) < 0.0);
    }

    #[test]
    fn test_rmsprop_step_updates_parameters() {
        let device = Default::default();
        let mut optimizer = MetaOptimizer::<TestBackend>::with_rmsprop(0.01, 1, 0.9, 1e-8);

        let mut params = vec![Tensor::<TestBackend, 2>::zeros([1, 1], &device)];
        let grads = vec![Some(Tensor::<TestBackend, 2>::ones([1, 1], &device))];

        optimizer.step(&mut params, &grads);
        assert!(scalar_f32(&params[0]) < 0.0);
    }
}
