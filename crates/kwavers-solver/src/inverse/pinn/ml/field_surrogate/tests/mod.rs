//! Phase C-1 / C-2 tests for the parameterised field-surrogate PINN.
//!
//! Partitioned by domain:
//! - `config` — `ParamFieldPINNConfig` validation tests (C-1).
//! - `network` — architecture construction and forward-pass shape tests (C-1).
//! - `infer` — `infer_grid` integration and invariance tests (C-1).
//! - `training` — trainer step, convergence, and Helmholtz-path tests (C-2).

mod config;
mod dynamic_tanh;
mod infer;
mod network;
mod sampling;
mod target_transform;
mod training;

/// Shared type alias for all sub-modules.
pub(super) type B = coeus_core::MoiraiBackend;
