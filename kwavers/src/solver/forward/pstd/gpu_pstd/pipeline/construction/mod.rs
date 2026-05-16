//! `GpuPstdSolver::new` — buffer allocation, shader compilation, pipeline build.
//!
//! SRP: changes when the buffer layout, shader path, push-constant size,
//! or pipeline entry-point names change.

mod kspace;
mod solver;

pub(super) use kspace::precompute_kspace_shifts;
