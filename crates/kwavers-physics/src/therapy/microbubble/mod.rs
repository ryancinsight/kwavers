//! Microbubble dynamics domain models.
//!
//! Entity and value-object models for therapeutic/contrast microbubbles: the
//! complete bubble state, the Marmottant shell model, and radiation/streaming
//! forces. The drug-payload value object (a therapy-delivery concern) lives in
//! the `kwavers-therapy` crate.

pub mod forces;
pub mod shell;
pub mod state;

pub use forces::{
    calculate_acoustic_streaming_velocity, calculate_drag_force, calculate_primary_bjerknes_force,
    calculate_primary_bjerknes_force_averaged, RadiationForce, StreamingVelocity,
};
pub use shell::{MarmottantShellProperties, ShellState};
pub use state::{MicrobubbleState, Position3D, Velocity3D};
