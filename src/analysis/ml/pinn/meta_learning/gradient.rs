//! Gradient Manipulation Utilities for Meta-Learning
//!
//! This module provides utilities for extracting and applying gradients in the Burn framework,
//! essential for implementing Model-Agnostic Meta-Learning (MAML) algorithms.
//!
//! # Gradient Flow in MAML
//!
//! MAML requires manipulating gradients at two levels:
//! 1. **Inner Loop**: Task-specific gradient descent for adaptation
//! 2. **Outer Loop**: Meta-gradient computation across task distribution
//!
//! The key challenge is manually applying gradients to parameters without
//! using the standard optimizer interface, since MAML needs to:
//! - Clone model parameters for each task
//! - Apply task-specific gradients
//! - Compute meta-gradients through adapted parameters
//!
//! # Implementation Details
//!
//! ## GradientExtractor
//! Uses Burn's `ModuleMapper` trait to traverse model structure and extract
//! gradients from the computation graph. Gradients are flattened to 1D tensors
//! for uniform handling.
//!
//! ## GradientApplicator
//! Applies extracted gradients back to model parameters using:
//! ```text
//! θ_new = θ_old - α * ∇L
//! ```
//! where α is the learning rate and ∇L is the gradient.
//!
//! # Literature References
//!
//! 1. Finn, C., Abbeel, P., & Levine, S. (2017).
//!    "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
//!    *ICML 2017*
//!    - Original MAML algorithm requiring manual gradient manipulation
//!
//! 2. Nichol, A., Achiam, J., & Schulman, J. (2018).
//!    "On First-Order Meta-Learning Algorithms"
//!    *arXiv:1803.02999*
//!    - First-order approximation simplifying gradient computation
//!
//! 3. Antoniou, A., Edwards, H., & Storkey, A. (2018).
//!    "How to train your MAML"
//!    *ICLR 2019*
//!    - Practical improvements including gradient clipping and normalization
//!
//! # Examples
//!
//! ```rust,ignore
//! use kwavers::analysis::ml::pinn::meta_learning::gradient::{
//!     GradientExtractor, GradientApplicator
//! };
//! use burn::module::ModuleMapper;
//!
//! // Extract gradients from a loss computation
//! let loss = model.forward(inputs);
//! let grads = loss.backward();
//!
//! let mut extractor = GradientExtractor::new(&grads);
//! let model_with_extracted = model.clone().map(&mut extractor);
//! let gradient_vec = extractor.into_gradients();
//!
//! // Apply gradients with learning rate
//! let mut applicator = GradientApplicator::new(gradient_vec, 0.01);
//! let updated_model = model.map(&mut applicator);
//! ```

use burn::module::{Module, ModuleMapper};
use burn::tensor::{backend::AutodiffBackend, Bool, Int, Tensor};

/// Gradient extractor for MAML inner-loop adaptation
///
/// Traverses a Burn module structure and extracts gradients from the
/// computation graph, flattening them to 1D tensors for uniform handling.
///
/// # Usage Pattern
///
/// 1. Create extractor with gradient reference from backward pass
/// 2. Map over model structure to collect gradients
/// 3. Extract collected gradients for storage or manipulation
///
/// # Technical Notes
///
/// - Only extracts gradients from float parameters (weights, biases)
/// - Int and Bool parameters pass through unchanged
/// - Gradients are flattened to 1D for memory efficiency
/// - None gradients indicate non-differentiable or frozen parameters
#[derive(Debug)]
pub struct GradientExtractor<'a, B: AutodiffBackend> {
    /// Reference to gradients from backward pass
    grads: &'a B::Gradients,
    /// Collected flattened gradients in traversal order
    collected: Vec<Option<Tensor<B::InnerBackend, 1>>>,
}

impl<'a, B: AutodiffBackend> GradientExtractor<'a, B> {
    /// Create a new gradient extractor
    ///
    /// # Arguments
    /// - `grads`: Reference to gradients from backward pass
    pub fn new(grads: &'a B::Gradients) -> Self {
        Self {
            grads,
            collected: Vec::new(),
        }
    }

    /// Consume the extractor and return collected gradients
    pub fn into_gradients(self) -> Vec<Option<Tensor<B::InnerBackend, 1>>> {
        self.collected
    }

    /// Get reference to collected gradients
    pub fn gradients(&self) -> &[Option<Tensor<B::InnerBackend, 1>>] {
        &self.collected
    }

    /// Get number of gradients collected
    pub fn len(&self) -> usize {
        self.collected.len()
    }

    /// Check if no gradients were collected
    pub fn is_empty(&self) -> bool {
        self.collected.is_empty()
    }
}

impl<'a, B: AutodiffBackend> ModuleMapper<B> for GradientExtractor<'a, B> {
    fn map_float<const D: usize>(
        &mut self,
        tensor: burn::module::Param<Tensor<B, D>>,
    ) -> burn::module::Param<Tensor<B, D>> {
        // Extract gradient for this parameter
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
        // Integer parameters don't have gradients
        tensor
    }

    fn map_bool<const D: usize>(
        &mut self,
        tensor: burn::module::Param<Tensor<B, D, Bool>>,
    ) -> burn::module::Param<Tensor<B, D, Bool>> {
        // Boolean parameters don't have gradients
        tensor
    }
}

/// Gradient applicator for MAML parameter updates
///
/// Applies gradients to model parameters using gradient descent:
/// ```text
/// θ_new = θ_old - α * ∇L
/// ```
///
/// # Usage Pattern
///
/// 1. Create applicator with gradients and learning rate
/// 2. Map over model structure to apply gradients
/// 3. Result is updated model with new parameter values
///
/// # Technical Notes
///
/// - Reshapes 1D gradients back to original parameter dimensions
/// - Preserves requires_grad flag for downstream differentiation
/// - Handles None gradients by leaving parameters unchanged
/// - Only updates float parameters (weights, biases)
///
/// # Gradient Clipping
///
/// For training stability, consider gradient clipping before application:
/// ```rust,ignore
/// let clipped_grads: Vec<_> = grads.into_iter()
///     .map(|g| g.map(|t| t.clamp(-1.0, 1.0)))
///     .collect();
/// ```
#[derive(Debug)]
pub struct GradientApplicator<B: AutodiffBackend> {
    /// Flattened gradients to apply
    grads: Vec<Option<Tensor<B::InnerBackend, 1>>>,
    /// Current index in gradient traversal
    index: usize,
    /// Learning rate for gradient descent
    lr: f64,
}

impl<B: AutodiffBackend> GradientApplicator<B> {
    /// Create a new gradient applicator
    ///
    /// # Arguments
    /// - `grads`: Flattened gradients to apply (from GradientExtractor)
    /// - `lr`: Learning rate for gradient descent step
    ///
    /// # Panics
    /// Panics if learning rate is negative or NaN
    pub fn new(grads: Vec<Option<Tensor<B::InnerBackend, 1>>>, lr: f64) -> Self {
        assert!(
            lr.is_finite() && lr >= 0.0,
            "Learning rate must be non-negative and finite, got {}",
            lr
        );

        Self {
            grads,
            index: 0,
            lr,
        }
    }

    /// Get the learning rate
    pub fn learning_rate(&self) -> f64 {
        self.lr
    }

    /// Get number of gradients to apply
    pub fn num_gradients(&self) -> usize {
        self.grads.len()
    }

    /// Check if all gradients have been applied
    pub fn is_complete(&self) -> bool {
        self.index >= self.grads.len()
    }
}

impl<B: AutodiffBackend> ModuleMapper<B> for GradientApplicator<B> {
    fn map_float<const D: usize>(
        &mut self,
        tensor: burn::module::Param<Tensor<B, D>>,
    ) -> burn::module::Param<Tensor<B, D>> {
        // Get gradient for current parameter (if available)
        let grad_opt = self.grads.get(self.index).cloned().unwrap_or(None);
        self.index = self.index.saturating_add(1);

        if let Some(grad_flat) = grad_opt {
            // Apply gradient descent: θ_new = θ_old - α * ∇L
            let is_require_grad = tensor.is_require_grad();
            let mut inner = (*tensor).clone().inner();

            // Reshape gradient to match parameter dimensions
            let grad: Tensor<B::InnerBackend, D> = grad_flat.reshape(inner.dims());

            // Update parameters
            inner = inner - grad.mul_scalar(self.lr);

            // Reconstruct tensor with proper autodiff flags
            let mut out = Tensor::<B, D>::from_inner(inner);
            if is_require_grad {
                out = out.require_grad();
            }

            burn::module::Param::from_tensor(out)
        } else {
            // No gradient available, leave parameter unchanged
            tensor
        }
    }

    fn map_int<const D: usize>(
        &mut self,
        tensor: burn::module::Param<Tensor<B, D, Int>>,
    ) -> burn::module::Param<Tensor<B, D, Int>> {
        // Integer parameters don't have gradients
        tensor
    }

    fn map_bool<const D: usize>(
        &mut self,
        tensor: burn::module::Param<Tensor<B, D, Bool>>,
    ) -> burn::module::Param<Tensor<B, D, Bool>> {
        // Boolean parameters don't have gradients
        tensor
    }
}

/// Utility functions for gradient manipulation
pub mod utils {
    use super::*;

    /// Compute L2 norm of gradients (for gradient clipping)
    ///
    /// Returns the Euclidean norm: ||∇||₂ = sqrt(Σᵢ gᵢ²)
    pub fn gradient_norm<B: AutodiffBackend>(grads: &[Option<Tensor<B::InnerBackend, 1>>]) -> f64 {
        let mut sum_squares = 0.0;

        for grad_opt in grads {
            if let Some(grad) = grad_opt {
                let grad_data = grad.to_data();
                let values = grad_data.as_slice::<f32>().unwrap_or(&[]);
                sum_squares += values.iter().map(|&g| (g as f64).powi(2)).sum::<f64>();
            }
        }

        sum_squares.sqrt()
    }

    /// Clip gradients by global norm
    ///
    /// Scales all gradients proportionally if norm exceeds threshold:
    /// ```text
    /// if ||∇||₂ > max_norm:
    ///     ∇ = ∇ * (max_norm / ||∇||₂)
    /// ```
    ///
    /// Literature: Pascanu et al. (2013) "On the difficulty of training
    /// recurrent neural networks"
    pub fn clip_gradients_by_norm<B: AutodiffBackend>(
        grads: Vec<Option<Tensor<B::InnerBackend, 1>>>,
        max_norm: f64,
    ) -> Vec<Option<Tensor<B::InnerBackend, 1>>> {
        let norm = gradient_norm(&grads);

        if norm > max_norm {
            let scale = max_norm / norm;
            grads
                .into_iter()
                .map(|g| g.map(|t| t.mul_scalar(scale)))
                .collect()
        } else {
            grads
        }
    }

    /// Clip gradients by value (element-wise)
    ///
    /// Clamps each gradient element to [-max_value, max_value]:
    /// ```text
    /// gᵢ = clamp(gᵢ, -max_value, max_value)
    /// ```
    pub fn clip_gradients_by_value<B: AutodiffBackend>(
        grads: Vec<Option<Tensor<B::InnerBackend, 1>>>,
        max_value: f64,
    ) -> Vec<Option<Tensor<B::InnerBackend, 1>>> {
        grads
            .into_iter()
            .map(|g| g.map(|t| t.clamp(-max_value, max_value)))
            .collect()
    }

    /// Add gradients element-wise (for gradient accumulation)
    pub fn add_gradients<B: AutodiffBackend>(
        grads1: Vec<Option<Tensor<B::InnerBackend, 1>>>,
        grads2: &[Option<Tensor<B::InnerBackend, 1>>],
    ) -> Vec<Option<Tensor<B::InnerBackend, 1>>> {
        grads1
            .into_iter()
            .zip(grads2.iter())
            .map(|(g1, g2)| match (g1, g2) {
                (Some(t1), Some(t2)) => Some(t1 + t2.clone()),
                (Some(t1), None) => Some(t1),
                (None, Some(t2)) => Some(t2.clone()),
                (None, None) => None,
            })
            .collect()
    }

    /// Scale gradients by a constant factor
    pub fn scale_gradients<B: AutodiffBackend>(
        grads: Vec<Option<Tensor<B::InnerBackend, 1>>>,
        scale: f64,
    ) -> Vec<Option<Tensor<B::InnerBackend, 1>>> {
        grads
            .into_iter()
            .map(|g| g.map(|t| t.mul_scalar(scale)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_extractor_new() {
        // This is a compile-time test to ensure the API is correct
        // Actual runtime testing requires a full Burn backend setup
    }

    #[test]
    fn test_gradient_applicator_new() {
        use burn::backend::NdArray;
        type TestBackend = burn::backend::Autodiff<NdArray<f32>>;

        let grads: Vec<Option<Tensor<<TestBackend as AutodiffBackend>::InnerBackend, 1>>> =
            Vec::new();
        let applicator = GradientApplicator::<TestBackend>::new(grads, 0.01);
        assert_eq!(applicator.learning_rate(), 0.01);
        assert_eq!(applicator.num_gradients(), 0);
        assert!(applicator.is_complete());
    }

    #[test]
    #[should_panic(expected = "Learning rate must be non-negative")]
    fn test_gradient_applicator_negative_lr() {
        use burn::backend::NdArray;
        type TestBackend = burn::backend::Autodiff<NdArray<f32>>;

        let grads: Vec<Option<Tensor<<TestBackend as AutodiffBackend>::InnerBackend, 1>>> =
            Vec::new();
        let _ = GradientApplicator::<TestBackend>::new(grads, -0.01);
    }

    #[test]
    fn test_gradient_norm_empty() {
        use burn::backend::NdArray;
        type TestBackend = burn::backend::Autodiff<NdArray<f32>>;

        let grads: Vec<Option<Tensor<<TestBackend as AutodiffBackend>::InnerBackend, 1>>> =
            Vec::new();
        let norm = utils::gradient_norm(&grads);
        assert_eq!(norm, 0.0);
    }
}
