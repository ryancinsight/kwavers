//! Orchestrator Initialization
//!
//! Initialization logic for therapy modality-specific subsystems.
//! Each modality requires specific initialization of hardware models,
//! control systems, and computational components.
//!
//! ## References
//!
//! - FDA 510(k) Guidance: Device-specific initialization requirements
//! - IEC 62359:2010: Safety parameter validation

pub mod lithotripsy;
pub mod modalities;

pub use lithotripsy::init_lithotripsy_simulator;
pub use modalities::{
    init_cavitation_controller, init_ceus_system, init_chemical_model, init_transcranial_system,
};
