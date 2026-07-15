//! Property-Based Gradient Validation Tests
//!
//! This module validates the correctness of automatic differentiation in the PINN
//! elastic wave solver by comparing autodiff gradients against finite difference
//! approximations.
//!
//! # Mathematical Foundation
//!
//! For a function f: ℝⁿ → ℝ, the gradient is:
//!
//! ```text
//! ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
//! ```
//!
//! Finite difference approximations:
//! - Forward: ∂f/∂x ≈ (f(x+h) - f(x)) / h
//! - Central: ∂f/∂x ≈ (f(x+h) - f(x-h)) / (2h)
//!
//! For second derivatives:
//! ```text
//! ∂²f/∂x² ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
//! ```
//!
//! # Validation Strategy
//!
//! 1. **First-order gradients**: Compare autodiff ∂u/∂x against central differences
//! 2. **Second-order gradients**: Validate ∂²u/∂x² for wave equation
//! 3. **Mixed derivatives**: Check ∂²u/∂x∂y for stress tensor
//! 4. **Property tests**: Validate across random input domains
//!
//! # Acceptance Criteria
//!
//! - Relative error < 1e-3 for first derivatives (h=1e-5)
//! - Relative error < 1e-2 for second derivatives (h=1e-4)
//! - Consistent across batch sizes and spatial domains

#[cfg(test)]
mod tests;
