//! Bubble Dynamics Module
//!
//! This module provides the core bubble dynamics calculations that are used by:
//! - Mechanics: for cavitation damage and erosion
//! - Optics: for sonoluminescence light emission

//! - Chemistry: for ROS generation and sonochemistry
//!
//! Based on the Keller-Miksis equation and extended models from literature

pub mod adaptive_integration; // NEW: Adaptive time-stepping for stiff ODEs
pub mod bjerknes_forces; // NEW: Bjerknes forces for bubble-bubble interactions
pub mod bubble_field;
pub mod bubble_state;
pub mod cavitation_control;
pub mod encapsulated; // NEW: Encapsulated bubbles with shell dynamics (Church, Marmottant)
pub mod energy_balance;
pub mod epstein_plesset; // NEW: Epstein-Plesset stability theorem implementation
pub mod gilmore; // Gilmore equation for violent collapse // NEW: Comprehensive energy balance model
pub mod imex_integration;
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
