//! Acoustic Physics Module
//!
//! Implements acoustic wave propagation, bubble dynamics, medical imaging,
//! therapeutic ultrasound, and acoustic mechanics with proper layer separation.
//!
//! ## Module Organization
//!
//! - **conservation**: Energy, mass, and momentum conservation validation
//! - **state**: Physics state container for field management
//! - **traits**: Core acoustic model interfaces and behaviors
//! - **bubble_dynamics**: Cavitation, bubble oscillation, microbubble models
//! - **imaging**: Medical imaging modalities (CEUS, elastography, ultrasound)
//! - **mechanics**: Acoustic effects on materials and tissues
//! - **therapy**: Therapeutic ultrasound applications (HIFU, drug delivery)
//! - **analysis**: Beam pattern and pressure field analysis
//! - **analytical**: Validation solutions and propagation models
//! - **wave_propagation**: Wave equations and heterogeneous media
//!
//! ## Explicit Re-exports
//!
//! This module uses explicit re-exports instead of wildcards to maintain
//! a clear, predictable public API and prevent namespace pollution.

pub mod analysis;
pub mod analytical;
pub mod bubble_dynamics;
pub mod conservation;
pub mod functional;
pub mod imaging;
pub mod mechanics;
pub mod skull;
pub mod state;
pub mod therapy;
pub mod traits;
pub mod transcranial;
pub mod wave_propagation;

// ============================================================================
// EXPLICIT RE-EXPORTS (Core Acoustic Physics API)
// ============================================================================

// Conservation validation (physical correctness checks)
pub use conservation::{
    validate_conservation, validate_energy_conservation, validate_mass_conservation,
    validate_momentum_conservation, ConservationMetrics,
};

// Physics state management
pub use state::{FieldView, FieldViewMut, HasPhysicsState, PhysicsState};

// Acoustic model traits and interfaces
pub use traits::{
    AcousticScatteringModelTrait, AcousticWaveModel, CavitationModelBehavior, ChemicalModelTrait,
    HeterogeneityModelTrait, LightDiffusionModelTrait, StreamingModelTrait, ThermalModelTrait,
};
