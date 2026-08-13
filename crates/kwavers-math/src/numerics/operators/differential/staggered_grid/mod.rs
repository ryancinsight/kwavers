//! Staggered grid finite difference operator (Yee scheme).
//!
//! SRP split:
//! - `operator`  — struct + constructor + `DifferentialOperator` impl
//! - `forward`   — `apply_forward_{x,y,z}[_into]` methods
//! - `backward`  — `apply_backward_{x,y,z}[_into]` methods
//! - `coefficients` — derived half-grid stencil weights to arbitrary even order
//! - `divergence` — Yee divergence: the adjoint of the forward difference

mod backward;
pub mod coefficients;
mod divergence;
mod forward;
mod operator;
#[cfg(test)]
mod tests;

pub use coefficients::{
    central_first_derivative_coefficients, staggered_first_derivative_coefficients, MAX_HALF_ORDER,
};
pub use operator::StaggeredGridOperator;
