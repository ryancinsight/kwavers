//! Application layer: PINN solver orchestration for 3D wave equation
//!
//! This module implements the high-level solver that orchestrates training and inference
//! for the 3D wave equation PINN. It combines the network, optimizer, and physics-informed
//! loss computation into a unified workflow.
//!
//! ## Solver Responsibilities
//!
//! - Orchestrate training loop with physics-informed loss
//! - Manage collocation point generation for PDE residual
//! - Coordinate network, optimizer, and geometry
//! - Provide prediction interface for trained models
//!
//! ## Training Workflow
//!
//! 1. Convert training data to tensors
//! 2. Generate collocation points for PDE residual
//! 3. For each epoch:
//!    - Compute physics-informed loss (data + PDE + BC + IC)
//!    - Backpropagate gradients
//!    - Update network parameters via optimizer
//! 4. Return training metrics
//!
//! ## Loss Components
//!
//! - **Data loss**: MSE between predictions and observations
//! - **PDE loss**: MSE of wave equation residual at collocation points
//! - **BC loss**: Boundary condition violations
//! - **IC loss**: Initial condition violations

pub mod collocation;
pub mod core;
pub mod diagnostics;
pub mod ics;
pub mod losses;
pub mod training;

pub use core::PinnWave3D;
