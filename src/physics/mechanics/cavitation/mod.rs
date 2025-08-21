//! # Cavitation Modeling Module
//!
//! This module focuses on the mechanical effects of cavitation, specifically:
//! - Material damage from bubble collapse
//! - Erosion and pitting
//! - Fatigue accumulation
//!
//! The actual bubble dynamics are handled by the bubble_dynamics module,
//! while this module calculates the mechanical consequences.
//!
//! - `core`: Provides the CavitationModel for field-based cavitation tracking
//! - `damage`: Calculates mechanical damage, erosion, and material fatigue from
//!   cavitation bubble collapse impacts.

pub mod core;
pub mod damage;

pub use core::CavitationModel;
pub use damage::{cavitation_intensity, CavitationDamage, DamageParameters, MaterialProperties};
