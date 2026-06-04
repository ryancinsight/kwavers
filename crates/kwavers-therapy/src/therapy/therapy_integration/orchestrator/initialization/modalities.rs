//! Modality-specific initialization for CEUS, transcranial, chemical, and cavitation subsystems.

use kwavers_core::constants::REFERENCE_FREQUENCY_HZ;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_domain::medium::Medium;
use kwavers_physics::cavitation_control::{ControlStrategy, FeedbackConfig, FeedbackController};
use kwavers_physics::chemistry::ChemicalModel;
use kwavers_physics::transcranial::TranscranialAberrationCorrection;
use kwavers_simulation::imaging::ceus::ContrastEnhancedUltrasound;

use super::super::super::config::{TherapyIntegrationModality, TherapySessionConfig};

/// Initialize CEUS system for microbubble therapy
///
/// Creates a contrast-enhanced ultrasound system with clinical microbubble parameters.
/// Typical clinical contrast agent concentrations are 1-10 million bubbles/mL.
///
/// # References
///
/// - Stride & Coussios (2010): "Nucleation, mapping and control of cavitation for drug delivery"
/// - FDA Guidance for Ultrasound Contrast Agents
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn init_ceus_system(
    grid: &Grid,
    medium: &dyn Medium,
) -> KwaversResult<ContrastEnhancedUltrasound> {
    let bubble_concentration = 1e6; // 1 million bubbles/mL (typical clinical dose)
    let bubble_size = 2.5; // 2.5 μm mean diameter (typical for clinical contrast agents)
    ContrastEnhancedUltrasound::new(grid, medium, bubble_concentration, bubble_size)
}

/// Initialize transcranial system
///
/// Creates a transcranial aberration correction system for focused ultrasound through the skull.
///
/// # References
///
/// - Aubry et al. (2003): "Experimental demonstration of noninvasive transskull adaptive focusing"
/// - Marsac et al. (2012): "MR-guided adaptive focusing of therapeutic ultrasound beams"
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn init_transcranial_system(
    _config: &TherapySessionConfig,
    grid: &Grid,
) -> KwaversResult<TranscranialAberrationCorrection> {
    TranscranialAberrationCorrection::new(grid)
}

/// Initialize chemical model for sonodynamic therapy
///
/// Creates a chemical reaction model for sonodynamic therapy applications.
///
/// # References
///
/// - Umemura et al. (1996): "Sonodynamic therapy: a novel approach to cancer treatment"
/// - Suslick (1990): "Sonochemistry"
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn init_chemical_model(grid: &Grid) -> KwaversResult<ChemicalModel> {
    ChemicalModel::new(grid, true, true)
}

/// Initialize cavitation controller for histotripsy/oncotripsy
///
/// Creates a feedback controller for cavitation-based therapy modalities.
///
/// # References
///
/// - Hall et al. (2010): "Histotripsy: minimally invasive tissue ablation using cavitation"
/// - Xu et al. (2016): "Oncotripsy: targeted cancer therapy using tumor-specific cavitation"
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
/// # Panics
/// - Panics if an internal precondition is violated.
///
pub fn init_cavitation_controller(
    config: &TherapySessionConfig,
) -> KwaversResult<FeedbackController> {
    let feedback_config = match config.primary_modality {
        TherapyIntegrationModality::Histotripsy => FeedbackConfig {
            strategy: ControlStrategy::AmplitudeOnly,
            target_intensity: 0.8,
            max_amplitude: 1.0,
            min_amplitude: 0.0,
            response_time: 0.001,
            safety_factor: 0.5,
            enable_adaptive: true,
        },
        TherapyIntegrationModality::Oncotripsy => FeedbackConfig {
            strategy: ControlStrategy::AmplitudeOnly,
            target_intensity: 0.6,
            max_amplitude: 1.0,
            min_amplitude: 0.0,
            response_time: 0.002,
            safety_factor: 0.7,
            enable_adaptive: true,
        },
        _ => unreachable!("Cavitation controller only for histotripsy/oncotripsy"),
    };
    Ok(FeedbackController::new(
        feedback_config,
        REFERENCE_FREQUENCY_HZ,
        1000.0,
    ))
}
