//! Staggered grid finite difference operator (Yee scheme).
//!
//! SRP split:
//! - `operator`  — struct + constructor + `DifferentialOperator` impl
//! - `forward`   — `apply_forward_{x,y,z}[_into]` methods
//! - `backward`  — `apply_backward_{x,y,z}[_into]` methods

mod backward;
mod forward;
mod operator;
#[cfg(test)]
mod tests;

pub use operator::StaggeredGridOperator;
