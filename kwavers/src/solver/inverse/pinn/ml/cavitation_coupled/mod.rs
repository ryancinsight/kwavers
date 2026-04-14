//! Cavitation-Acoustic Coupled Physics Domain for PINN
//!
//! This module implements the coupling between acoustic wave propagation and cavitation
//! bubble dynamics. The acoustic pressure field drives bubble oscillations, while bubble
//! dynamics can modify the acoustic field through scattering and nonlinear effects.
//!
//! ## Mathematical Formulation
//!
//! The coupled system solves:
//! - Acoustic wave equation: ∂²p/∂t² = c²∇²p + nonlinear terms + bubble scattering
//! - Bubble dynamics: Keller-Miksis equation with acoustic forcing
//! - Coupling: p_acoustic drives bubble wall acceleration
//!
//! ## Coupling Types
//!
//! 1. **Weak Coupling**: Acoustic field drives bubble dynamics (one-way)
//! 2. **Strong Coupling**: Mutual interaction with scattering and nonlinear effects
//! 3. **Multi-bubble Coupling**: Collective bubble effects and Bjerknes forces

use crate::physics::bubble_dynamics::{BubbleParameters, BubbleState, KellerMiksisModel};
use crate::solver::inverse::pinn::ml::physics::{
    BoundaryComponent, BoundaryConditionSpec, BoundaryPosition, CouplingInterface, CouplingType,
    InitialConditionSpec, PhysicsDomain, PhysicsLossWeights, PhysicsParameters,
    PhysicsValidationMetric,
};
use burn::prelude::ElementConversion;
use burn::tensor::{backend::AutodiffBackend, Tensor};
use num_complex::Complex;
use std::collections::HashMap;

pub mod config;
pub mod domain;
pub mod mie_scattering;

pub use config::*;
pub use domain::*;
pub use mie_scattering::*;
