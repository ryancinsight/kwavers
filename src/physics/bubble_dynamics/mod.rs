//! Bubble Dynamics Module
//!
//! This module provides the core bubble dynamics calculations that are used by:
//! - Mechanics: for cavitation damage and erosion
//! - Optics: for sonoluminescence light emission
//! - Chemistry: for ROS generation and sonochemistry
//!
//! Based on the Keller-Miksis equation and extended models from literature

pub mod bubble_state;
pub mod rayleigh_plesset;
pub mod bubble_field;
pub mod interactions;
pub mod thermodynamics;
pub mod imex_integration;
pub mod adaptive_integration;  // NEW: Adaptive time-stepping for stiff ODEs
pub mod energy_balance;  // NEW: Comprehensive energy balance model
pub mod units;  // NEW: Unit-safe types using uom crate

pub use bubble_state::{BubbleState, BubbleParameters, GasSpecies};
pub use rayleigh_plesset::{RayleighPlessetSolver, KellerMiksisModel};
pub use bubble_field::{BubbleField, BubbleCloud, BubbleStateFields};
pub use interactions::{BubbleInteractions, BjerknesForce, CollectiveEffects};
pub use imex_integration::{
    BubbleIMEXIntegrator, BubbleIMEXConfig, integrate_bubble_dynamics_imex
};
pub use adaptive_integration::{
    AdaptiveBubbleIntegrator, AdaptiveBubbleConfig, 
    integrate_bubble_dynamics_adaptive, IntegrationStatistics
};
