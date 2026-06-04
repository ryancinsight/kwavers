//! Therapy Integration Orchestrator
//!
//! Main orchestrator for clinical therapy sessions, integrating multiple ultrasound therapy
//! modalities with comprehensive safety monitoring and multi-physics coupling.
//!
//! ## Supported Modalities
//!
//! - **HIFU**: High-intensity focused ultrasound for tumor ablation
//! - **LIFU**: Low-intensity focused ultrasound for drug delivery
//! - **Histotripsy**: Mechanical tissue ablation via cavitation
//! - **Oncotripsy**: Tumor-specific histotripsy
//! - **Sonodynamic**: ROS-mediated cancer therapy
//! - **Lithotripsy**: Kidney stone fragmentation
//! - **Transcranial**: Focused ultrasound through the skull
//! - **CEUS**: Contrast-enhanced ultrasound therapy
//!
//! ## Architecture
//!
//! The orchestrator follows a composition pattern, delegating specialized functionality
//! to focused submodules while maintaining a clean API for therapy execution.
//!
//! ## References
//!
//! - AAPM TG-166: "Quality assurance for ultrasound-guided interventions"
//! - IEC 62359:2010: "Ultrasonics - Field characterization"
//! - FDA 510(k) Guidance: "Ultrasound Devices"

use crate::therapy::lithotripsy::LithotripsySimulator;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_physics::cavitation_control::FeedbackController;
use kwavers_physics::chemistry::ChemicalModel;
use kwavers_physics::transcranial::TranscranialAberrationCorrection;
use kwavers_simulation::imaging::ceus::ContrastEnhancedUltrasound;

use super::acoustic::AcousticWaveSolver;
use super::config::TherapySessionConfig;
use super::intensity_tracker::IntensityTracker;
use super::safety_controller::SafetyController;
use super::state::TherapySessionState;

// Submodule declarations
pub mod cavitation;
pub mod chemical;
pub mod execution;
pub mod initialization;
pub mod lithotripsy;
pub mod microbubble;
pub mod safety;

mod methods;
#[cfg(test)]
mod tests;

/// Therapy integration orchestrator
///
/// Coordinates multiple therapy modalities with safety monitoring and multi-physics coupling.
/// Manages the complete therapy session lifecycle from initialization through execution
/// to completion and safety validation.
#[derive(Debug)]
pub struct TherapyIntegrationOrchestrator {
    /// Session configuration
    pub(super) config: TherapySessionConfig,
    /// Computational grid
    pub(super) grid: Grid,
    /// Medium properties
    pub(super) medium: Box<dyn Medium>,
    /// Acoustic wave solver
    pub(super) _acoustic_solver: AcousticWaveSolver,
    /// CEUS system (for microbubble therapy)
    pub(super) ceus_system: Option<ContrastEnhancedUltrasound>,
    /// Transcranial correction system
    pub(super) _transcranial_system: Option<TranscranialAberrationCorrection>,
    /// Chemical model (for sonodynamic therapy)
    pub(super) chemical_model: Option<ChemicalModel>,
    /// Cavitation detector and controller
    pub(super) cavitation_controller: Option<FeedbackController>,
    /// Lithotripsy simulator
    pub(super) lithotripsy_simulator: Option<LithotripsySimulator>,
    /// Real-time safety controller for therapy enforcement
    pub(super) safety_controller: SafetyController,
    /// Real-time intensity tracker for acoustic monitoring
    pub(super) intensity_tracker: IntensityTracker,
    /// Current session state
    pub(super) session_state: TherapySessionState,
}
