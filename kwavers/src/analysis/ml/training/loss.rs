//! Physics-informed loss components and optimizer types.

use ndarray::Array2;

/// Physics loss component
///
/// Enforces wave equation and acoustic constraints:
/// - Reciprocity: Forward and reverse paths should be equal
/// - Coherence: Adjacent elements should have similar phases
/// - Sparsity: Suppress noise via L1 regularization on weights
#[derive(Debug, Clone)]
pub struct PhysicsLoss {
    /// Weight for reciprocity constraint
    pub reciprocity_weight: f64,
    /// Weight for coherence constraint
    pub coherence_weight: f64,
    /// Weight for sparsity (L1 regularization)
    pub sparsity_weight: f64,
}

impl Default for PhysicsLoss {
    fn default() -> Self {
        Self {
            reciprocity_weight: 0.5,
            coherence_weight: 0.3,
            sparsity_weight: 0.2,
        }
    }
}

impl PhysicsLoss {
    /// Compute reciprocity constraint violation
    ///
    /// For a reciprocal system: response(A→B) = response(B→A)
    /// Violation = ||forward - reverse||²
    #[must_use]
    pub fn reciprocity_violation(forward: &Array2<f64>, reverse: &Array2<f64>) -> f64 {
        if forward.dim() != reverse.dim() {
            return f64::INFINITY;
        }
        if forward.is_empty() {
            return 0.0;
        }
        let diff = forward - reverse;
        diff.iter().map(|x| x * x).sum::<f64>() / (forward.len() as f64)
    }

    /// Compute coherence constraint (phase continuity)
    ///
    /// Adjacent elements should have similar phases (continuous wavefront)
    /// Violation = sum_i |phase(i+1) - phase(i)|
    #[must_use]
    pub fn coherence_violation(phases: &Array2<f64>) -> f64 {
        if phases.is_empty() || phases.dim().0 < 2 {
            return 0.0;
        }
        // Shortest-arc magnitude of each adjacent-element phase difference
        // (SSOT wrap; see `math::signal::wrap_to_pi`). Correct for any gradient
        // magnitude, unlike a single π-fold of |Δφ|.
        let mut violation = 0.0;
        for i in 0..phases.dim().0 - 1 {
            for j in 0..phases.dim().1 {
                let normalized =
                    crate::math::signal::wrap_to_pi(phases[[i + 1, j]] - phases[[i, j]]).abs();
                violation += normalized;
            }
        }
        violation / phases.len() as f64
    }

    /// Compute sparsity constraint (L1 regularization on weights)
    #[must_use]
    pub fn sparsity_violation(weights: &Array2<f64>) -> f64 {
        if weights.is_empty() {
            return 0.0;
        }
        weights.iter().map(|w| w.abs()).sum::<f64>() / weights.len() as f64
    }
}

/// Gradient-based optimizer
///
/// # Literature
///
/// - Kingma & Ba (2015) "Adam: A Method for Stochastic Optimization"
#[derive(Debug, Clone, Copy)]
pub enum Optimizer {
    /// Stochastic Gradient Descent
    SGD,
    /// SGD with Momentum
    Momentum { beta: f64 },
    /// Adam (Adaptive Moment Estimation)
    Adam {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    /// RMSprop (Root Mean Square Propagation)
    RMSprop { beta: f64, epsilon: f64 },
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}
