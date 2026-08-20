//! Monolithic residual subsystem.
//!
//! This directory is the internal Newton-Krylov physics layer for
//! [`super::MonolithicCoupler`].  It separates the mathematical
//! responsibilities that were previously co-located in one file:
//!
//! - residual assembly: `F(u) = u - u_prev - dt * R(u)`;
//! - Jacobian-free vector products: `Jv ≈ [F(u + eps v) - F(u)] / eps`;
//! - adaptive line search over residual-evaluated Newton candidates;
//! - the Athena `LinearOperator` seam that presents those products to the
//!   Krylov solve.
//!
//! The split introduces no runtime dispatch or wrapper API.  The physics
//! modules add inherent methods to `MonolithicCoupler` and the operator seam
//! is a borrowing view over it, so call sites keep static dispatch and the
//! compiler monomorphizes the same concrete `Array3<f64>` kernels.

mod compute;
mod jacobian_operator;
mod jvp;
mod line_search;

pub(in crate::multiphysics::monolithic) use jacobian_operator::JacobianOperator;

#[cfg(test)]
mod tests;
