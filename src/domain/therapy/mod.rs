//! Therapy Domain Models
//!
//! Domain entities and value objects for therapeutic ultrasound applications.
//!
//! ## Bounded Contexts
//!
//! This module contains domain models for various therapeutic ultrasound
//! modalities organized into bounded contexts:
//!
//! - **Microbubble Dynamics**: Contrast agents and drug delivery vehicles
//! - **Focused Ultrasound**: HIFU ablation and thermal therapy (future)
//! - **Neuromodulation**: Brain stimulation (future)
//! - **Thrombolysis**: Clot dissolution (future)
//!
//! ## Domain-Driven Design
//!
//! The therapy domain follows DDD principles:
//!
//! 1. **Ubiquitous Language**: Medical/physical terminology shared across team
//! 2. **Bounded Contexts**: Clear boundaries between therapy modalities
//! 3. **Aggregates**: Entities with identity and lifecycle (e.g., MicrobubbleState)
//! 4. **Value Objects**: Immutable descriptors (e.g., Position3D, RadiationForce)
//! 5. **Domain Events**: State transitions (e.g., shell rupture, cavitation)
//!
//! ## Architecture
//!
//! ```text
//! domain/therapy/
//! ├── microbubble/        - Microbubble dynamics bounded context
//! │   ├── state.rs        - Entity: Complete bubble state
//! │   ├── shell.rs        - Value object: Marmottant shell model
//! │   ├── drug_payload.rs - Value object: Drug release kinetics
//! │   └── forces.rs       - Value object: Radiation forces
//! └── mod.rs              - This file
//! ```
//!
//! ## Usage
//!
//! ```rust,no_run
//! use kwavers::domain::therapy::microbubble::{MicrobubbleState, Position3D};
//!
//! // Create therapeutic microbubble
//! let position = Position3D::new(0.01, 0.02, 0.03);
//! let bubble = MicrobubbleState::drug_loaded(2.0, 50.0, position).unwrap();
//!
//! println!("Loaded drug mass: {:.2e} kg", bubble.drug_mass());
//! ```

pub mod microbubble;

// Re-export commonly used types
pub use microbubble::{
    calculate_primary_bjerknes_force, DrugLoadingMode, DrugPayload,
    MarmottantShellProperties, MicrobubbleState, Position3D, RadiationForce, ShellState,
    Velocity3D,
};
