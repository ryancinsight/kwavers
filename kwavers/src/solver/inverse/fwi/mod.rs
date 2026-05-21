//! Full-waveform inversion method families.
//!
//! Frequency-domain and time-domain FWI use different state equations,
//! discretizations, and adjoint contracts. This module makes that split explicit:
//!
//! - `frequency_domain`: Helmholtz-domain complex pressure inversion.
//!   Forward-operator selection via the [`frequency_domain::HelmholtzForwardOperator`]
//!   trait (single-scatter Born / dense CBS / spectral CBS today; future impls
//!   plug in additively).
//! - `time_domain`: FDTD-driven acoustic full-waveform inversion driver
//!   (`FwiProcessor`, `FwiGeometry`) plus the L2 / adjoint-state primitives
//!   shared with reverse-time migration. Unified-dispatcher migration (so the
//!   forward solver becomes selectable as FDTD / PSTD / Hybrid via
//!   `SolverType` and `SimulationSolverFactory`) tracked in backlog T15.

pub mod frequency_domain;
pub mod time_domain;
