//! Bubble dynamics module
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

pub use bubble_state::{BubbleState, BubbleParameters, GasSpecies};
pub use rayleigh_plesset::{RayleighPlessetSolver, KellerMiksisModel};
pub use bubble_field::{BubbleField, BubbleCloud, BubbleStateFields};
pub use interactions::{BubbleInteractions, BjerknesForce, CollectiveEffects};