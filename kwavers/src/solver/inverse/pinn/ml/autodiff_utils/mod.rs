//! Burn autodiff utilities for PINN gradient computation.
//!
//! This module centralizes Burn autodiff derivative helpers and keeps each
//! derivative family in a separate file. The public API remains the function
//! set exposed by the former monolithic module.
//!
//! # Gradient Computation Pattern
//!
//! ```rust,ignore
//! let input = input.require_grad();
//! let output = model.forward(input.clone());
//! let grads = output.backward();
//! let grad_tensor = input.grad(&grads);
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
