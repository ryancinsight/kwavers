//! Microbubble Domain Entities and Value Objects
//!
//! This module contains the core domain model for therapeutic microbubbles
//! used in ultrasound-mediated drug delivery and contrast-enhanced imaging.
//!
//! ## Domain Model
//!
//! The microbubble domain is structured according to Domain-Driven Design (DDD)
//! principles with clear separation of:
//!
//! - **Entities**: Objects with identity and lifecycle (e.g., `MicrobubbleState`)
//! - **Value Objects**: Immutable objects defined by attributes (e.g., `Position3D`, `RadiationForce`)
//! - **Domain Services**: Operations that don't naturally belong to entities (in application layer)
//!
//! ## Module Structure
//!
//! ```text
//! microbubble/
//! ├── state.rs          - MicrobubbleState entity (geometric, dynamic, thermodynamic)
//! ├── shell.rs          - Marmottant shell model (elasticity, state transitions)
//! ├── drug_payload.rs   - Drug encapsulation and release kinetics
//! └── forces.rs         - Radiation forces (Bjerknes, streaming, drag)
//! ```
//!
//! ## Bounded Context
//!
//! This module represents the **Microbubble Dynamics** bounded context within
//! the larger Kwavers therapeutic ultrasound domain. It has well-defined
//! interfaces with:
//!
//! - **Acoustic Field Context**: Receives pressure fields, gradients
//! - **Therapy Planning Context**: Provides bubble states for safety analysis
//! - **Drug Delivery Context**: Exposes release kinetics and concentrations
//!
//! ## Ubiquitous Language
//!
//! - **Microbubble**: Gas-filled microsphere with lipid shell (1-10 μm diameter)
//! - **Equilibrium Radius (R₀)**: Bubble radius at rest without ultrasound
//! - **Wall Velocity (Ṙ)**: Rate of change of bubble radius
//! - **Marmottant Model**: Shell mechanics with buckling and rupture
//! - **Bjerknes Force**: Radiation force from pressure gradient
//! - **Cavitation**: Violent bubble collapse (inertial cavitation)
//! - **Acoustic Streaming**: Steady flow induced by bubble oscillation
//!
//! ## Mathematical Foundations
//!
//! The domain model is based on:
//!
//! 1. **Keller-Miksis Equation**: Compressible bubble dynamics
//! 2. **Marmottant Shell Model**: Nonlinear shell mechanics
//! 3. **Bjerknes Forces**: Radiation pressure and gradient forces
//! 4. **First-Order Kinetics**: Drug release modeling
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use kwavers::domain::therapy::microbubble::{
//!     MicrobubbleState, Position3D, MarmottantShellProperties,
//!     calculate_primary_bjerknes_force,
//! };
//!
//! // Create microbubble at position
//! let position = Position3D::new(0.01, 0.02, 0.05); // meters
//! let mut bubble = MicrobubbleState::sono_vue(position).unwrap();
//!
//! // Calculate radiation force from pressure gradient
//! let grad_p = (1e5, 0.0, 0.0); // Pa/m
//! let force = calculate_primary_bjerknes_force(
//!     bubble.radius,
//!     bubble.radius_equilibrium,
//!     grad_p,
//! ).unwrap();
//!
//! println!("Radiation force: {:.3e} N", force.magnitude());
//! ```
//!
//! ## References
//!
//! - Keller & Miksis (1980): "Bubble oscillations of large amplitude"
//! - Marmottant et al. (2005): "A model for large amplitude oscillations of coated bubbles"
//! - Stride & Coussios (2010): "Nucleation, mapping and control of cavitation for drug delivery"
//! - Ferrara et al. (2007): "Ultrasound microbubble contrast agents"

pub mod drug_payload;
pub mod forces;
pub mod shell;
pub mod state;

// Re-export commonly used types for convenience
pub use drug_payload::{DrugLoadingMode, DrugPayload};
pub use forces::{
    calculate_acoustic_streaming_velocity, calculate_drag_force, calculate_primary_bjerknes_force,
    calculate_primary_bjerknes_force_averaged, RadiationForce, StreamingVelocity,
};
pub use shell::{MarmottantShellProperties, ShellState};
pub use state::{MicrobubbleState, Position3D, Velocity3D};
