//! Bubble Dynamics Module
//!
//! This module provides the core bubble dynamics calculations that are used by:
//! - Mechanics: for cavitation damage and erosion
//! - Optics: for sonoluminescence light emission
//! - Chemistry: for ROS generation and sonochemistry
//!
//! Based on the Keller-Miksis equation and extended models from literature

pub mod adaptive_integration; // NEW: Adaptive time-stepping for stiff ODEs
pub mod bubble_field;
pub mod bubble_state;
pub mod energy_balance; // NEW: Comprehensive energy balance model
pub mod imex_integration;
pub mod interactions;
pub mod rayleigh_plesset;
pub mod thermodynamics;
pub mod units; // NEW: Unit-safe types using uom crate

pub use adaptive_integration::{
    integrate_bubble_dynamics_adaptive, AdaptiveBubbleConfig, AdaptiveBubbleIntegrator,
    IntegrationStatistics,
};
pub use bubble_field::{BubbleCloud, BubbleField, BubbleStateFields};
pub use bubble_state::{BubbleParameters, BubbleState, GasSpecies};
pub use imex_integration::{
    integrate_bubble_dynamics_imex, BubbleIMEXConfig, BubbleIMEXIntegrator,
};
pub use interactions::{BjerknesForce, BubbleInteractions, CollectiveEffects};
pub use rayleigh_plesset::{KellerMiksisModel, RayleighPlessetSolver};
