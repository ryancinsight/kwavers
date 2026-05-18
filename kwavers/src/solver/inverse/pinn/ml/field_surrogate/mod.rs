//! Parameterised field-surrogate PINN — `(x, y, z, f0, pnp) → (p_min,
//! p_max, p_rms)`.
//!
//! Extends the existing `burn_wave_equation_3d` infrastructure with a
//! **static** field-surrogate network whose output is the per-voxel
//! time-integrated pressure-statistics tuple, parameterised by the
//! source frequency and target peak rarefactional pressure. Where the
//! transient wave-equation PINN learns `u(x,y,z,t)` for one fixed
//! `(f0, pnp)` scenario, this surrogate learns the *envelope shape*
//! across the `(f0, pnp)` sweep so a single forward pass replaces a
//! full PSTD solve in treatment planners.
//!
//! ## Pipeline
//!
//! 1. Generate a kernel sweep with
//!    [`crate::physics::field_surrogate::KernelCube`] (data ground-
//!    truth) covering the operating envelope.
//! 2. Train this network supervised on the cube voxels — input
//!    `(x, y, z, f0, pnp)`, target `(p_min, p_max, p_rms)`.
//! 3. At planning time, evaluate the trained network on the planner
//!    grid for any `(f0, pnp)` in or out of the cube. ~ms-scale
//!    inference vs minutes for PSTD.
//!
//! ## Phase C-1 (this version)
//!
//! Module skeleton + 5D-input MLP network + grid-query forward
//! inference. Training loop and pyo3 binding are queued for C-2.

#![cfg(feature = "pinn")]

pub mod config;
pub mod dynamic_tanh;
pub mod forward;
pub mod network;
pub mod optimizer;
pub mod sampler;
pub mod target_transform;
pub mod training;
pub mod types;

#[cfg(test)]
mod tests;

pub use config::ParamFieldPINNConfig;
pub use dynamic_tanh::DynamicTanh;
pub use forward::{infer_grid, GridQueryParams};
pub use network::ParamFieldPINNNetwork;
pub use optimizer::ParamFieldOptimizer;
pub use sampler::KernelCubeSampler;
pub use target_transform::{OutputTransforms, TargetTransform};
pub use training::{
    FieldSurrogateTrainingConfig, ParamFieldPINNTrainer, StepMetrics, SurrogateTrainingMetrics,
    TrainingBatch,
};
pub use types::{CoordHalves, OutputScales, ParamRanges, SamplingMode};
