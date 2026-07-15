//! PDE residual computation for the 2D elastic wave equation.
//!
//! ```text
//! ρ ∂²u/∂t² = ∇·σ    where σ = λ tr(ε) I + 2μ ε,  ε = ∇_s u
//! ```
//!
//! The physics and finite-difference numerics live in
//! [`crate::inverse::pinn::ml::autodiff_utils::elastic`], shared with every
//! other elastic-wave PINN in this crate. This module adapts
//! `ElasticPINN2D`'s three-separate-`Var` `forward(x, y, t)` signature to
//! that shared utility's single-tensor `forward_fn` convention — see
//! [`pipeline::compute_elastic_wave_pde_residual`].

pub mod pipeline;

#[cfg(test)]
mod tests;

pub use pipeline::compute_elastic_wave_pde_residual;
