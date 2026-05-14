//! Monolithic residual subsystem.
//!
//! This directory is the internal Newton-Krylov physics layer for
//! [`MonolithicCoupler`](super::coupler::MonolithicCoupler).  It separates the
//! three mathematical responsibilities that were previously co-located in one
//! file:
//!
//! - residual assembly: `F(u) = u - u_prev - dt * R(u)`;
//! - Jacobian-free vector products: `Jv ≈ [F(u + eps v) - F(u)] / eps`;
//! - adaptive line search over residual-evaluated Newton candidates.
//!
//! The split introduces no runtime dispatch or wrapper API.  Each child module
//! adds inherent methods to `MonolithicCoupler`, so call sites keep static
//! dispatch and the compiler monomorphizes the same concrete `Array3<f64>`
//! kernels.

mod compute;
mod jvp;
mod line_search;

#[cfg(test)]
mod tests;
