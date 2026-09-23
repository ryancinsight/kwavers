//! Elastic stress tensor and its divergence.
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
//! Every derivative is leto's fourth-order central operator
//! (`leto_ops::FiniteDifference3D::central_fourth_order`, ADR 128):
//! ```text
//! ∂f/∂x ≈ (-f[i+2] + 8f[i+1] - 8f[i-1] + f[i-2]) / (12·Δx)
//! ```
//! in the interior, second-order central one point from a wall, first-order
//! one-sided at the wall, and zero along a singleton axis.
//!
//! ## Algorithm
//!
//! `stress_divergence` uses a two-pass scheme:
//! - Pass 1: sweep the displacement derivatives and assemble all 6 stress
//!   components from them and the spatially-varying Lamé parameters.
//! - Pass 2: sweep the stress derivatives and sum them into the 3-component
//!   divergence `(∇·σ)_x`, `(∇·σ)_y`, `(∇·σ)_z`.

mod divergence;
mod slabs;
#[cfg(test)]
mod tests;

pub use divergence::{stress_divergence, stress_divergence_into};
pub(crate) use divergence::{stress_divergence_plane_strain_into, DensityScale};
pub(crate) use slabs::stress_acceleration_into;

// The slab sweep probe times slab heights the production rule does not pick.
#[cfg(test)]
pub(crate) use slabs::stress_acceleration_in_slabs;
