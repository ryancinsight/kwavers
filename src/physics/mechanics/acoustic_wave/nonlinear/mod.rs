//! # Nonlinear Acoustic Wave Simulation Module
//!
//! This module provides the structures and implementations for simulating nonlinear acoustic
//! wave propagation. It refactors the `NonlinearWave` solver into several focused submodules:
//!
//! - `config`: Contains the `NonlinearWave` struct definition, its constructor (`new`),
//!   and configuration methods (e.g., `set_nonlinearity_scaling`, `set_max_pressure`).
//!   This is the primary entry point for creating and configuring a nonlinear wave solver.
//! - `core`: Houses the main `update_wave` method, which is responsible for advancing the
//!   simulation by a single time step. This includes applying source terms, calculating
//!   nonlinear effects, performing linear propagation in k-space, and combining these components.
//! - `stability`: Provides methods for checking and enforcing simulation stability, such as
//!   `check_stability` (which evaluates CFL conditions and field value integrity) and
//!   `clamp_pressure` (which limits pressure values to prevent numerical divergence).
//! - `performance`: Includes methods for monitoring and reporting the computational performance
//!   of the solver, like `report_performance`, which logs time spent in various parts of the
//!   simulation loop.
//! - `helpers`: Contains utility functions used by the other modules, such as
//!   `calculate_phase_factor` for k-space propagation calculations.
//!
//! The `NonlinearWave` struct itself is re-exported from the `config` submodule for convenient access.
// src/physics/mechanics/acoustic_wave/nonlinear/mod.rs
pub mod config;
pub mod core;
pub mod helpers;
pub mod performance;
pub mod stability;

// Re-export NonlinearWave. The struct definition will live in config.rs
pub use config::NonlinearWave;
