//! # Cavitation Modeling Module
//!
//! This module is responsible for simulating the dynamics of cavitation bubbles and
//! their associated physical effects within a medium subjected to acoustic fields.
//! It is refactored into several submodules to organize the different aspects of
//! cavitation modeling:
//!
//! - `model`: Defines the core `CavitationModel` struct, which holds the state
//!   (e.g., bubble radius, velocity, temperature) and parameters of the cavitation field.
//!   It also includes the constructor and basic accessor methods for the model.
//! - `dynamics`: Contains the implementations for the equations of motion governing
//!   bubble dynamics, primarily focusing on calculating bubble wall acceleration
//!   (e.g., via Rayleigh-Plesset type equations) and integrating these to update
//!   bubble radius and velocity over time.
//! - `effects`: Implements the physical consequences of bubble activity, such as
//!   the acoustic pressure changes due to bubble volume pulsations and scattering,
//!   and the emission of light (sonoluminescence) during violent bubble collapses.
//! - `core`: Houses the main `update_cavitation` method, which orchestrates the
//!   entire process of advancing the cavitation simulation by a single time step.
//!   It calls methods from `dynamics` and `effects` to update the bubble states
//!   and calculate their impact on the surrounding fields.
//!
//! The `CavitationModel` struct is re-exported from the `model` submodule for
//! convenient access from other parts of the simulation framework.

pub mod core;
pub mod dynamics;
pub mod effects;
pub mod model;

pub use model::CavitationModel;

pub mod trait_impls; // Added to include the trait implementations

// Note: All constants previously at the top level of this file 
// (STEFAN_BOLTZMANN, MIN_RADIUS, MAX_RADIUS, MAX_VELOCITY, MAX_ACCELERATION)
// have been moved into the respective submodules where they are used:
// - STEFAN_BOLTZMANN to effects.rs (as pub(crate) const)
// - General physical limit constants (MIN_RADIUS_MODEL_DEFAULT, etc.) to model.rs (as pub(crate) const)
// - Specific operational constants that were local within the original update_cavitation 
//   (now in core.rs) remain local to that function.
// The original comment "// physics/mechanics/cavitation/mod.rs" and its associated use statements
// are no longer needed here as the actual code has been moved.
// Each submodule (core, dynamics, effects, model) is responsible for its own imports.
