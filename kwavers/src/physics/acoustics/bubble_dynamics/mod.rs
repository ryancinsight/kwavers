//! Bubble Dynamics Module
//!
//! This module provides the core bubble dynamics calculations that are used by:
//! - Mechanics: for cavitation damage and erosion
//! - Optics: for sonoluminescence light emission
//! - Chemistry: for ROS generation and sonochemistry
//!
//! Based on the Keller-Miksis equation and extended models from literature.
//!
//! # ARCHITECTURE NOTE (SOC / SRP debt)
//!
//! The `adaptive_integration` and `imex_integration` sub-modules implement ODE
//! **time-stepping** logic (adaptive Runge-Kutta, IMEX schemes), which
//! architecturally belongs in the **solver layer** (e.g., `solver/forward/ode/`),
//! not the physics layer.  The physics layer should only define equations of
//! motion (Keller-Miksis, Rayleigh-Plesset, etc.).
//!
//! Tracked for future refactor — see `docs/IMPLEMENTATION_ROADMAP.md` §Layer Boundaries.
//!
//! ## Migration Path
//! 1. Create `solver/forward/ode/` with `AdaptiveRkSolver<E: BubbleOde>` trait
//! 2. Move `adaptive_integration` and `imex_integration` to `solver/forward/ode/`
//! 3. Physics layer keeps only the equation structs implementing `BubbleOde`
//! 4. Update all callers to the new solver-layer path

pub mod adaptive_integration; // NEW: Adaptive time-stepping for stiff ODEs
pub mod bjerknes_forces; // NEW: Bjerknes forces for bubble-bubble interactions
pub mod bubble_field;
pub mod bubble_state;
pub mod cavitation_control;
pub mod encapsulated; // NEW: Encapsulated bubbles with shell dynamics (Church, Marmottant)
pub mod energy;
pub mod epstein_plesset; // NEW: Epstein-Plesset stability theorem implementation
pub mod gilmore; // Gilmore equation for violent collapse // NEW: Comprehensive energy balance model
pub mod imex_integration;

pub use energy::{update_temperature_energy_balance, EnergyBalanceCalculator};
pub mod integration; // NEW: Stable integration utilities extracted from monolithic file
pub mod interactions;
pub mod keller_miksis; // NEW: Extracted Keller-Miksis solver for modularity
pub mod rayleigh_plesset;
pub mod thermodynamics;
pub mod units; // NEW: Unit-safe types using uom crate

pub use adaptive_integration::{
    integrate_bubble_dynamics_adaptive, AdaptiveBubbleConfig, AdaptiveBubbleIntegrator,
    IntegrationStatistics,
};
pub use bjerknes_forces::{BjerknesCalculator, BjerknesConfig, BjerknesForce, InteractionType}; // NEW: Bubble-bubble interaction forces
pub use bubble_field::{BubbleCloud, BubbleField, BubbleStateFields};
pub use bubble_state::{BubbleParameters, BubbleState, GasSpecies};
pub use cavitation_control::{
    CavitationMetrics, ControlOutput, ControlStrategy, FeedbackConfig, FeedbackController,
};
pub use encapsulated::{ChurchModel, MarmottantModel, ShellProperties}; // NEW: Encapsulated bubble models
pub use epstein_plesset::EpsteinPlessetStabilitySolver; // NEW: Epstein-Plesset stability analysis
pub use imex_integration::{
    integrate_bubble_dynamics_imex, BubbleIMEXConfig, BubbleIMEXIntegrator,
};
pub use integration::integrate_bubble_dynamics_stable; // NEW: Extracted integration utilities
pub use interactions::{
    BjerknesForce as InteractionBjerknesForce, BubbleInteractions, CollectiveEffects,
};
pub use keller_miksis::KellerMiksisModel; // NEW: Modular Keller-Miksis solver
pub use rayleigh_plesset::RayleighPlessetSolver;
