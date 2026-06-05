//! Therapeutic-ultrasound domain models.
//!
//! Domain entities and value objects for therapeutic ultrasound: microbubble
//! dynamics (state, Marmottant shell, radiation forces) and the therapy
//! modality/mechanism/parameter/metric descriptors. Drug-payload delivery
//! models live in the `kwavers-therapy` crate, which depends on this one.

pub mod microbubble;
pub mod types;

pub use microbubble::{
    calculate_primary_bjerknes_force, MarmottantShellProperties, MicrobubbleState, Position3D,
    RadiationForce, ShellState, Velocity3D,
};
pub use types::{
    DomainTherapyMechanism, DomainTherapyModality, DomainTherapyParameters, DomainTreatmentMetrics,
};
