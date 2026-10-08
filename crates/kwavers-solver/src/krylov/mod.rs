//! Krylov linear solves, delegated to Athena.
//!
//! Athena owns the Krylov recurrences, the linear-operator and preconditioner
//! seams, and convergence policy for the Atlas stack (Atlas ADR 0033). This
//! module holds no recurrence. It carries only what is genuinely kwavers':
//!
//! - [`GMRESConfig`], the restart and tolerance vocabulary kwavers' solver
//!   configuration already speaks, translated into an Athena
//!   [`ConvergencePolicy`](athena_core::ConvergencePolicy);
//! - [`GmresConvergenceInfo`], the convergence summary kwavers' coupled-step
//!   reports carry, read out of an Athena
//!   [`SolveReport`](athena_core::SolveReport);
//! - [`KrylovWorkspace`], the one runtime-to-compile-time restart bridge the
//!   stack carries, owned by Athena's Leto backend (Atlas ADR 0062): the
//!   ladder of `Gmres` instantiations and its dispatch live in
//!   `athena-leto`, and this module re-exports it under the name kwavers'
//!   call sites already speak.
//!
//! It is the single home for Krylov solves in this crate: the dense boundary
//! element system ([`crate::forward::bem`]) and the matrix-free Newton-Krylov
//! coupler ([`crate::multiphysics::monolithic`]) both drive Athena through it.

mod cg;
mod config;
mod failure;
mod report;

#[cfg(test)]
mod tests;

pub use athena_leto::KrylovWorkspace;
pub use cg::{solve_cg, CpuBackend, SliceOperator, SlicePreconditioner};
pub use config::GMRESConfig;
pub use report::GmresConvergenceInfo;

pub(crate) use config::policy;
pub(crate) use failure::{backend_failure, solve_failure};
pub(crate) use report::convergence_failure;
