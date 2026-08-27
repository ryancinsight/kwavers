//! `coeus_autograd` utilities for PINN gradient computation.
//!
//! This module centralizes derivative helpers over a generic
//! `Fn(&Var<f32,B>) -> Var<f32,B>` forward pass and keeps each
//! derivative family in a separate file.
//!
//! # Gradient Computation Pattern
//!
//! ```rust,ignore
//! let input_grad = Var::new(input.clone(), true);
//! let output = model_forward(&input_grad);
//! output.backward();
//! let grad_tensor = input_grad.grad();
//! ```
//!
//! # Mathematical Specifications
//!
//! For a two-dimensional displacement field `u(t, x, y) = [u_x, u_y]`, the
//! helpers compute first time derivatives, second time derivatives, spatial
//! gradients, divergence, Laplacian, strain, and elastic-wave residual terms:
//!
//! ```text
//! rho d²u/dt² = (lambda + 2mu) grad(div u) + mu laplacian(u)
//! ```

use coeus_autograd::Var;
use kwavers_core::error::KwaversResult;

/// Output contract accepted by autodiff finite-difference utilities.
///
/// Pure forward closures return `Var`; model-backed closures may return
/// `KwaversResult<Var>` so backend and input-contract failures remain visible.
pub trait ForwardOutput<B: coeus_ops::BackendOps<f32> + Default>: Sized {
    /// Convert the forward result into the solver error contract.
    fn into_forward_result(self) -> KwaversResult<Var<f32, B>>;
}

impl<B: coeus_ops::BackendOps<f32> + Default> ForwardOutput<B> for Var<f32, B> {
    fn into_forward_result(self) -> KwaversResult<Var<f32, B>> {
        Ok(self)
    }
}

impl<B: coeus_ops::BackendOps<f32> + Default> ForwardOutput<B> for KwaversResult<Var<f32, B>> {
    fn into_forward_result(self) -> KwaversResult<Var<f32, B>> {
        self
    }
}

/// Step size for the second-order central stencils in this module, derived
/// from the precision they evaluate in.
///
/// A second central difference carries two errors that move in opposite
/// directions with `h`: truncation, of order `h^2 * f4 / 12` where `f4` is
/// the fourth derivative, and round-off, of order `4 * eps * |f| / h^2`,
/// the latter because the stencil subtracts nearly equal values. Their sum
/// is minimised at `h = (48 * eps)^(1/4)`, which for `f32`'s
/// `eps = 1.19e-7` is `1.9e-2`, leaving a residual of order
/// `sqrt(eps) = 3.5e-4`.
///
/// These stencils used `1e-4`, which is the corresponding optimum for
/// `f64` -- `(48 * 2.2e-16)^(1/4)` is `1.1e-4`. Against `f32` it puts the
/// round-off floor at `4 * eps / h^2 = 47.7`, so the stencils returned that
/// number rather than a derivative, for any input at any point. It went
/// unnoticed because the only network exercising them had every weight set
/// to 1.0 by `coeus_nn::Linear::new` (coeus ADR 0067), which made the
/// differences vanish identically and took the noise with them.
pub(crate) const SECOND_ORDER_STEP: f32 = 1.9e-2;

mod elastic;
mod second_order;
mod spatial;
mod time;

pub use elastic::{compute_elastic_wave_residual_2d, compute_strain_tensor_2d};
pub use second_order::{
    compute_gradient_of_divergence_2d, compute_laplacian_2d, compute_second_derivative_2d,
};
pub use spatial::{compute_divergence_2d, compute_spatial_gradient_2d};
pub use time::{compute_second_time_derivative, compute_time_derivative};

#[cfg(test)]
mod tests;
