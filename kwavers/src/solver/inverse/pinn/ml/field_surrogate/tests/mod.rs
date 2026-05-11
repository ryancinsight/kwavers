//! Phase C-1 / C-2 tests for the parameterised field-surrogate PINN.
//!
//! Partitioned by domain:
//! - `config` — `ParamFieldPINNConfig` validation tests (C-1).
//! - `network` — architecture construction and forward-pass shape tests (C-1).
//! - `infer` — `infer_grid` integration and invariance tests (C-1).
//! - `training` — trainer step, convergence, and Helmholtz-path tests (C-2).

mod config;
mod infer;
mod network;
mod sampling;
mod training;

/// Shared type aliases for all sub-modules.
pub(super) type B = burn::backend::NdArray<f32>;
pub(super) type AB = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
