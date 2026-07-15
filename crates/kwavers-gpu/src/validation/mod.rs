//! GPU validation: differential CPU-vs-GPU equivalence checks.
//!
//! These live with the GPU backend (not in the algorithm crates) because this
//! is where concrete GPU providers are linked against CPU references from
//! `kwavers_solver`. The FDTD equivalence runner currently reports the GPU side
//! as unavailable until a real provider-generic Leto/Hephaestus FDTD trait
//! implementation is wired; it does not compare the CPU solver against itself.

#[cfg(feature = "gpu")]
pub mod gpu_cpu_equivalence;
