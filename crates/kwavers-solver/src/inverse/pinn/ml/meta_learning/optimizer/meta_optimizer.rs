//! MetaOptimizer struct and outer-loop update methods.
//!
//! Implements SGD, Momentum, Adam, and RMSProp update rules for MAML outer-loop.
//!
//! # Outer-Loop Optimization
//!
//! ```text
//! θ ← θ - β∇_θ Σ_τ L_τ(U_τ(θ))
//! ```
//!
//! # Literature References
//!
//! 1. Finn, C., Abbeel, P., & Levine, S. (2017). MAML. *ICML 2017*
//! 2. Kingma, D. P., & Ba, J. (2014). Adam. *arXiv:1412.6980*
//! 3. Antoniou, A., Edwards, H., & Storkey, A. (2018). How to train your MAML. *ICLR 2019*

use burn::tensor::{backend::AutodiffBackend, Tensor};

/// Meta-optimizer for outer-loop parameter updates.
///
/// Supports SGD, Momentum, Adam, and RMSProp update modes.
///
/// # Adam update rule
/// ```text
/// m ← β₁*m + (1-β₁)*∇L
/// v ← β₂*v + (1-β₂)*∇L²
/// m̂ ← m/(1-β₁ᵗ)
/// v̂ ← v/(1-β₂ᵗ)
/// θ ← θ - α*m̂/(√v̂ + ε)
/// ```
#[derive(Debug)]
pub struct MetaOptimizer<B: AutodiffBackend> {
    /// Outer-loop learning rate.
    lr: f64,

    /// Optimizer mode.
    mode: MetaOptimizerMode,

    /// Momentum parameter (0.0 = no momentum, 0.9 = typical).
    pub(crate) _momentum: Option<f64>,

    /// Adam beta1 parameter (first moment decay).
    pub(crate) _beta1: f64,

    /// Adam beta2 parameter (second moment decay).
    pub(crate) _beta2: f64,

    /// Adam epsilon for numerical stability.
    pub(crate) _epsilon: f64,

    /// Iteration count for bias correction.
    pub(crate) _iteration_count: usize,

    /// First moment estimates (for Adam/momentum).
    pub(crate) _m: Vec<Option<Tensor<B, 2>>>,

    /// Second moment estimates (for Adam/RMSProp).
    pub(crate) _v: Vec<Option<Tensor<B, 2>>>,
}

/// Optimization algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaOptimizerMode {
    Sgd,
    Momentum,
    Adam,
    RmsProp,
}

impl<B: AutodiffBackend> MetaOptimizer<B> {
    /// Create a new meta-optimizer with SGD mode.
    ///
    /// # Panics
    /// Panics if `lr` is non-positive or non-finite.
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

    /// Create meta-optimizer with Momentum mode.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
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

    /// Create meta-optimizer with Adam hyperparameters.
    /// # Panics
    /// - Panics if assertion fails: `Beta1 must be in (0, 1), got {}`.
    /// - Panics if assertion fails: `Beta2 must be in (0, 1), got {}`.
    /// - Panics if assertion fails: `Epsilon must be positive, got {}`.
    ///
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

    /// Create meta-optimizer with RMSProp hyperparameters.
    /// # Panics
    /// - Panics if assertion fails: `Decay must be in (0, 1), got {}`.
    /// - Panics if assertion fails: `Epsilon must be positive, got {}`.
    ///
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

    /// Perform one optimization step over `params` using `gradients`.
    ///
    /// `None` gradient entries leave the corresponding parameter unchanged.
    ///
    /// # SGD update
    /// ```text
    /// θ ← θ - α∇L
    /// ```
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
                            m_prev
                                .mul_scalar(beta1)
                                .add(g.clone().mul_scalar(1.0 - beta1))
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

                        let denom = v.clone().sqrt().add_scalar(eps);
                        *param = param
                            .clone()
                            .sub((g.clone() / denom).mul_scalar(self.lr as f32));

                        self._v[idx] = Some(v);
                    }
                }
            }
        }
    }

    /// Return the current learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.lr
    }

    /// Set a new learning rate (for LR scheduling).
    ///
    /// # Panics
    /// Panics if `lr` is non-positive or non-finite.
    pub fn set_learning_rate(&mut self, lr: f64) {
        assert!(
            lr > 0.0 && lr.is_finite(),
            "Learning rate must be positive and finite, got {}",
            lr
        );
        self.lr = lr;
    }

    /// Return the number of optimization steps performed.
    pub fn iteration_count(&self) -> usize {
        self._iteration_count
    }

    /// Reset all moment buffers and the iteration counter.
    pub fn reset(&mut self) {
        self._iteration_count = 0;
        self._m.iter_mut().for_each(|m| *m = None);
        self._v.iter_mut().for_each(|v| *v = None);
    }

    /// Multiply the current learning rate by `factor`.
    ///
    /// # Panics
    /// Panics if `factor` is outside `(0, 1]`.
    pub fn decay_learning_rate(&mut self, factor: f64) {
        assert!(
            factor > 0.0 && factor <= 1.0,
            "Decay factor must be in (0, 1], got {}",
            factor
        );
        self.lr *= factor;
    }
}
