//! Finite-difference utilities for elastic stress tensor computation.
//!
//! ## Mathematical Background
//!
//! The elastic wave equation in displacement form:
//! ```text
//! ρ ∂²u/∂t² = ∇·σ + f
//! ```
//!
//! where the isotropic stress tensor (Hooke's law) is:
//! ```text
//! σxx = (λ+2μ) εxx + λ(εyy+εzz)
//! σyy = (λ+2μ) εyy + λ(εxx+εzz)
//! σzz = (λ+2μ) εzz + λ(εxx+εyy)
//! σxy = σyx = μ(∂ux/∂y + ∂uy/∂x)
//! σxz = σzx = μ(∂ux/∂z + ∂uz/∂x)
//! σyz = σzy = μ(∂uy/∂z + ∂uz/∂y)
//! ```
//!
//! ## Numerical Method
//!
//! Fourth-order accurate centered finite difference for interior points:
//! ```text
//! ∂f/∂x ≈ (-f[i+2] + 8f[i+1] - 8f[i-1] + f[i-2]) / (12·Δx)
//! ```
//! Second-order one-sided stencils at boundaries (|i - boundary| < 2).
//!
//! ## Algorithm
//!
//! `stress_divergence` uses a two-pass scheme:
//! - Pass 1: compute all 6 stress components at every grid point from
//!   displacement gradients and spatially-varying Lamé parameters.
//! - Pass 2: differentiate the stress arrays to produce the 3-component
//!   divergence `(∇·σ)_x`, `(∇·σ)_y`, `(∇·σ)_z`.

mod divergence;
mod fd_stencils;
#[cfg(test)]
mod tests;

pub use divergence::{stress_divergence, stress_divergence_into};
pub use fd_stencils::{fd1_x, fd1_y, fd1_z};
