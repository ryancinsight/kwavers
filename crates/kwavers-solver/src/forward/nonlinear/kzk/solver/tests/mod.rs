//! Value-semantic regression tests for the KZK solver.
//!
//! Split by concern:
//! - [`creation`]     — solver construction and defaults
//! - [`beam`]         — Gaussian beam propagation (Tier 1 fast + Tier 3 comprehensive)
//! - [`conservation`] — energy/momentum conservation diagnostics
//! - [`solve_api`]    — `solve(n)` API invariants (zero steps, counter, bounds, parity)

mod beam;
mod conservation;
mod creation;
mod solve_api;
