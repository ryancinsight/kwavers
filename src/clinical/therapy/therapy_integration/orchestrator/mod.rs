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

use crate::clinical::therapy::lithotripsy::LithotripsySimulator;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::physics::cavitation_control::FeedbackController;
use crate::physics::chemistry::ChemicalModel;
use crate::physics::transcranial::TranscranialAberrationCorrection;
use crate::simulation::imaging::ceus::ContrastEnhancedUltrasound;

use super::acoustic::AcousticWaveSolver;
use super::config::{TherapyModality, TherapySessionConfig};
use super::state::{SafetyMetrics, SafetyStatus, TherapySessionState};

// Submodule declarations
pub mod cavitation;
pub mod chemical;
pub mod execution;
pub mod initialization;
pub mod lithotripsy;
pub mod microbubble;
pub mod safety;

use ndarray::Array3;

/// Therapy integration orchestrator
///
/// Coordinates multiple therapy modalities with safety monitoring and multi-physics coupling.
/// Manages the complete therapy session lifecycle from initialization through execution
/// to completion and safety validation.
#[derive(Debug)]
pub struct TherapyIntegrationOrchestrator {
    /// Session configuration
    config: TherapySessionConfig,
    /// Computational grid
    grid: Grid,
    /// Medium properties
    medium: Box<dyn Medium>,
    /// Acoustic wave solver
    _acoustic_solver: AcousticWaveSolver,
    /// CEUS system (for microbubble therapy)
    ceus_system: Option<ContrastEnhancedUltrasound>,
    /// Transcranial correction system
    _transcranial_system: Option<TranscranialAberrationCorrection>,
    /// Chemical model (for sonodynamic therapy)
    chemical_model: Option<ChemicalModel>,
    /// Cavitation detector and controller
    cavitation_controller: Option<FeedbackController>,
    /// Lithotripsy simulator
    lithotripsy_simulator: Option<LithotripsySimulator>,
    /// Current session state
    session_state: TherapySessionState,
}

impl TherapyIntegrationOrchestrator {
    /// Create new therapy integration orchestrator
    ///
    /// Initializes the orchestrator with configuration, grid, and medium properties.
    /// Automatically initializes modality-specific subsystems based on the configuration.
    ///
    /// # Arguments
    ///
    /// - `config`: Therapy session configuration
    /// - `grid`: Computational grid
    /// - `medium`: Acoustic medium properties
    ///
    /// # Returns
    ///
    /// Initialized orchestrator ready for therapy execution
    ///
    /// # Errors
    ///
    /// Returns error if any subsystem initialization fails
    pub fn new(
        config: TherapySessionConfig,
        grid: Grid,
        medium: Box<dyn Medium>,
    ) -> KwaversResult<Self> {
        // Initialize acoustic solver
        let acoustic_solver = AcousticWaveSolver::new(&grid, &*medium)?;

        // Initialize modality-specific systems
        let ceus_system = if config.primary_modality == TherapyModality::Microbubble
            || config
                .secondary_modalities
                .contains(&TherapyModality::Microbubble)
        {
            Some(initialization::init_ceus_system(&grid, &*medium)?)
        } else {
            None
        };

        let transcranial_system = if config.primary_modality == TherapyModality::Transcranial
            || config
                .secondary_modalities
                .contains(&TherapyModality::Transcranial)
        {
            Some(initialization::init_transcranial_system(&config, &grid)?)
        } else {
            None
        };

        let chemical_model = if config.primary_modality == TherapyModality::Sonodynamic
            || config
                .secondary_modalities
                .contains(&TherapyModality::Sonodynamic)
        {
            Some(initialization::init_chemical_model(&grid)?)
        } else {
            None
        };

        let cavitation_controller = if config.primary_modality == TherapyModality::Histotripsy
            || config.primary_modality == TherapyModality::Oncotripsy
            || config
                .secondary_modalities
                .contains(&TherapyModality::Histotripsy)
            || config
                .secondary_modalities
                .contains(&TherapyModality::Oncotripsy)
        {
            Some(initialization::init_cavitation_controller(&config)?)
        } else {
            None
        };

        let lithotripsy_simulator = if config.primary_modality == TherapyModality::Lithotripsy
            || config
                .secondary_modalities
                .contains(&TherapyModality::Lithotripsy)
        {
            Some(initialization::init_lithotripsy_simulator(&config, &grid)?)
        } else {
            None
        };

        let session_state = TherapySessionState {
            current_time: 0.0,
            progress: 0.0,
            acoustic_field: None,
            microbubble_concentration: None,
            cavitation_activity: None,
            chemical_concentrations: None,
            safety_metrics: SafetyMetrics {
                thermal_index: 0.0,
                mechanical_index: 0.0,
                cavitation_dose: 0.0,
                temperature_rise: Array3::zeros(grid.dimensions()),
            },
        };

        Ok(Self {
            config,
            grid,
            medium,
            _acoustic_solver: acoustic_solver,
            ceus_system,
            _transcranial_system: transcranial_system,
            chemical_model,
            cavitation_controller,
            lithotripsy_simulator,
            session_state,
        })
    }

    /// Execute therapy session step
    ///
    /// Advances the therapy session by one time step, including:
    /// - Acoustic field generation
    /// - Modality-specific updates (microbubbles, cavitation, chemistry, lithotripsy)
    /// - Safety metric calculation
    /// - Session state updates
    ///
    /// # Arguments
    ///
    /// - `dt`: Time step (s)
    ///
    /// # Returns
    ///
    /// Ok if step executed successfully, error otherwise
    ///
    /// # Side Effects
    ///
    /// Updates internal session state including time, progress, fields, and safety metrics
    pub fn execute_therapy_step(&mut self, dt: f64) -> KwaversResult<()> {
        // Update session time and progress
        self.session_state.current_time += dt;
        self.session_state.progress = self.session_state.current_time / self.config.duration;

        // Generate acoustic field
        let acoustic_field =
            execution::generate_acoustic_field(&self.grid, &self.config.acoustic_params)?;

        // Apply transcranial correction if enabled (future enhancement)
        let corrected_field = acoustic_field;

        // Update microbubble dynamics if enabled
        if let Some(ref mut ceus) = self.ceus_system {
            let concentration =
                microbubble::update_microbubble_dynamics(ceus, &corrected_field, dt)?;
            self.session_state.microbubble_concentration = concentration;
        }

        // Update cavitation activity if enabled
        if let Some(ref mut controller) = self.cavitation_controller {
            let cavitation_activity = cavitation::update_cavitation_control(
                controller,
                &corrected_field,
                &self.config.acoustic_params,
                dt,
            )?;
            self.session_state.cavitation_activity = Some(cavitation_activity);
        }

        // Update chemical reactions if enabled
        if let Some(ref mut chemistry) = self.chemical_model {
            let chemical_concentrations = chemical::update_chemical_reactions(
                chemistry,
                &corrected_field,
                self.session_state.cavitation_activity.as_ref(),
                &self.config.acoustic_params,
                &self.grid,
                &*self.medium,
                dt,
            )?;
            self.session_state.chemical_concentrations = Some(chemical_concentrations);
        }

        // Execute lithotripsy if enabled
        if let Some(ref mut simulator) = self.lithotripsy_simulator {
            let progress = lithotripsy::execute_lithotripsy_step(simulator, &corrected_field, dt)?;
            self.session_state.progress = progress;
        }

        // Update safety metrics
        safety::update_safety_metrics(
            &mut self.session_state.safety_metrics,
            &corrected_field,
            &self.config.acoustic_params,
            dt,
            self.session_state.cavitation_activity.as_ref(),
        )?;

        // Store current state
        self.session_state.acoustic_field = Some(corrected_field);

        Ok(())
    }

    /// Check safety limits
    ///
    /// Evaluates current safety metrics against configured limits.
    ///
    /// # Returns
    ///
    /// Safety status indicating whether therapy is safe or which limit was exceeded
    pub fn check_safety_limits(&self) -> SafetyStatus {
        safety::check_safety_limits(
            &self.session_state.safety_metrics,
            &self.config.safety_limits,
            self.session_state.current_time,
        )
    }

    /// Get current session state
    ///
    /// # Returns
    ///
    /// Reference to current therapy session state
    pub fn session_state(&self) -> &TherapySessionState {
        &self.session_state
    }

    /// Get session configuration
    ///
    /// # Returns
    ///
    /// Reference to therapy session configuration
    pub fn config(&self) -> &TherapySessionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clinical::therapy::therapy_integration::config::{
        AcousticTherapyParams, PatientParameters, SafetyLimits, TargetVolume, TissueType,
    };
    use crate::clinical::therapy::therapy_integration::tissue::TissuePropertyMap;
    use crate::domain::medium::homogeneous::HomogeneousMedium;

    #[test]
    fn test_therapy_orchestrator_creation() {
        let config = TherapySessionConfig {
            primary_modality: TherapyModality::Histotripsy,
            secondary_modalities: vec![TherapyModality::Microbubble],
            duration: 60.0,
            acoustic_params: AcousticTherapyParams {
                frequency: 1e6,
                pnp: 10e6,
                prf: 100.0,
                duty_cycle: 0.01,
                focal_depth: 0.05,
                treatment_volume: 1.0,
            },
            safety_limits: SafetyLimits {
                thermal_index_max: 6.0,
                mechanical_index_max: 1.9,
                cavitation_dose_max: 1000.0,
                max_treatment_time: 300.0,
            },
            patient_params: PatientParameters {
                skull_thickness: None,
                tissue_properties: TissuePropertyMap::liver((10, 10, 10)),
                target_volume: TargetVolume {
                    center: (0.05, 0.0, 0.0),
                    dimensions: (0.02, 0.02, 0.02),
                    tissue_type: TissueType::Liver,
                },
                risk_organs: vec![],
            },
            imaging_data_path: None,
        };

        let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.0, &grid);

        let orchestrator =
            TherapyIntegrationOrchestrator::new(config, grid, Box::new(medium.clone()));
        assert!(orchestrator.is_ok());

        let orchestrator = orchestrator.unwrap();
        assert_eq!(
            orchestrator.config().primary_modality,
            TherapyModality::Histotripsy
        );
        assert!(orchestrator.session_state().current_time < 1e-6);
    }

    #[test]
    #[ignore]
    fn test_therapy_step_execution() {
        let config = TherapySessionConfig {
            primary_modality: TherapyModality::Microbubble,
            secondary_modalities: vec![],
            duration: 10.0,
            acoustic_params: AcousticTherapyParams {
                frequency: 2e6,
                pnp: 1e6,
                prf: 100.0,
                duty_cycle: 0.1,
                focal_depth: 0.03,
                treatment_volume: 0.5,
            },
            safety_limits: SafetyLimits {
                thermal_index_max: 6.0,
                mechanical_index_max: 1.9,
                cavitation_dose_max: 1000.0,
                max_treatment_time: 300.0,
            },
            patient_params: PatientParameters {
                skull_thickness: None,
                tissue_properties: TissuePropertyMap::liver((16, 16, 16)),
                target_volume: TargetVolume {
                    center: (0.03, 0.0, 0.0),
                    dimensions: (0.01, 0.01, 0.01),
                    tissue_type: TissueType::Liver,
                },
                risk_organs: vec![],
            },
            imaging_data_path: None,
        };

        let grid = Grid::new(16, 16, 16, 0.002, 0.002, 0.002).unwrap();
        let medium = Box::new(HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.0, &grid));

        let mut orchestrator = TherapyIntegrationOrchestrator::new(config, grid, medium).unwrap();

        // Execute a few therapy steps
        let dt = 0.1;
        for _ in 0..5 {
            let result = orchestrator.execute_therapy_step(dt);
            assert!(result.is_ok());

            let safety_status = orchestrator.check_safety_limits();
            assert_eq!(safety_status, SafetyStatus::Safe);
        }

        assert!(orchestrator.session_state().current_time > 0.0);
        assert!(orchestrator.session_state().progress > 0.0);
        assert!(orchestrator.session_state().acoustic_field.is_some());
    }

    #[test]
    fn test_safety_limit_checking() {
        let config = TherapySessionConfig {
            primary_modality: TherapyModality::Transcranial,
            secondary_modalities: vec![],
            imaging_data_path: None,
            duration: 10.0,
            acoustic_params: AcousticTherapyParams {
                frequency: 0.5e6,
                pnp: 0.5e6,
                prf: 1.0,
                duty_cycle: 0.1,
                focal_depth: 0.05,
                treatment_volume: 1.0,
            },
            safety_limits: SafetyLimits {
                thermal_index_max: 0.5,
                mechanical_index_max: 1.9,
                cavitation_dose_max: 1000.0,
                max_treatment_time: 300.0,
            },
            patient_params: PatientParameters {
                skull_thickness: None,
                tissue_properties: TissuePropertyMap::liver((8, 8, 8)),
                target_volume: TargetVolume {
                    center: (0.05, 0.0, 0.0),
                    dimensions: (0.01, 0.01, 0.01),
                    tissue_type: TissueType::Brain,
                },
                risk_organs: vec![],
            },
        };

        let grid = Grid::new(8, 8, 8, 0.005, 0.005, 0.005).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.0, &grid);

        let mut orchestrator =
            TherapyIntegrationOrchestrator::new(config, grid, Box::new(medium.clone())).unwrap();

        let result = orchestrator.execute_therapy_step(1.0);
        assert!(result.is_ok());

        let safety_status = orchestrator.check_safety_limits();
        assert_eq!(safety_status, SafetyStatus::Safe);
    }
}
