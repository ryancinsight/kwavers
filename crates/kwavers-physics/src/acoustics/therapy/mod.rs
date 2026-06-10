//! Unified ultrasound therapy physics module
//!
//! This module consolidates all therapeutic ultrasound modalities including:
//! - HIFU (High-Intensity Focused Ultrasound)
//! - LIFU (Low-Intensity Focused Ultrasound)
//! - Histotripsy (mechanical tissue ablation)
//! - BBB (Blood-Brain Barrier) opening
//! - Sonodynamic therapy
//! - Sonoporation
//! - Microbubble-mediated therapies
//! - Sonogenetics (focused-ultrasound neuromodulation via mechanosensitive channels)
//!
//! ## Design Principles
//! - **SOLID**: Single responsibility per module
//! - **GRASP**: Modular organization under 200 lines per file
//! - **CUPID**: Composable therapy components
//! - **Zero-Cost**: Efficient abstractions

pub mod cavitation;
pub mod lithotripsy;
pub mod neuromodulation;
pub mod sonogenetics;

// Re-exports
pub use cavitation::{CavitationDetectionMethod, TherapyCavitationDetector};
// pub use metrics::TreatmentMetrics; // Moved to domain
// pub use modalities::{TherapyMechanism, TherapyModality}; // Moved to domain
// pub use parameters::TherapyParameters; // Moved to domain

// Note: TherapyCalculator has moved to crate::simulation::therapy::calculator
// Domain types are in kwavers_physics::therapy
