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
//! - [`KrylovWorkspace`], the bridge from a restart width chosen at runtime to
//!   Athena's compile-time `Gmres<B, RESTART>`.
//!
//! It is the single home for Krylov solves in this crate: the dense boundary
//! element system ([`crate::forward::bem`]) and the matrix-free Newton-Krylov
//! coupler ([`crate::multiphysics::monolithic`]) both drive Athena through it.

mod config;
mod report;
mod restart;

#[cfg(test)]
mod tests;

pub use config::GMRESConfig;
pub use report::GmresConvergenceInfo;
pub use restart::KrylovWorkspace;

pub(crate) use config::policy;
pub(crate) use report::convergence_failure;
