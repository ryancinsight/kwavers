//! `GpuPstdSolver::new` — buffer allocation, shader compilation, pipeline build.
//!
//! SRP: changes when the buffer layout, shader path, push-constant size,
//! or pipeline entry-point names change.

mod entry;
mod kspace;

pub(in crate::pstd_gpu) use entry::CheckedPlan;
pub(super) use entry::ConstructionError;
mod solver;
