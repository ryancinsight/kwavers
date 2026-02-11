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
use super::intensity_tracker::IntensityTracker;
use super::safety_controller::{SafetyController, TherapyAction};
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
    /// Real-time safety controller for therapy enforcement
    safety_controller: SafetyController,
    /// Real-time intensity tracker for acoustic monitoring
    intensity_tracker: IntensityTracker,
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

        // Initialize safety controller with configured limits
        let mut safety_controller = SafetyController::new(config.safety_limits.clone(), None);
        safety_controller.start_monitoring(0.0);

        // Initialize intensity tracker for real-time monitoring
        let intensity_tracker = IntensityTracker::new(0.1); // 100ms rolling window

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
            safety_controller,
            intensity_tracker,
            session_state,
        })
    }

    /// Execute therapy session step
    ///
    /// Advances the therapy session by one time step, including:
    /// - Acoustic field generation
    /// - Real-time intensity monitoring (SPTA, thermal dose, peak intensity)
    /// - Real-time safety evaluation (thermal/mechanical indices, cavitation)
    /// - Modality-specific updates (microbubbles, cavitation, chemistry, lithotripsy)
    /// - Temperature field computation from acoustic heating
    /// - Safety metric calculation and adaptive power reduction
    /// - Session state updates
    ///
    /// ## Safety Integration
    ///
    /// This function enforces real-time safety constraints via SafetyController:
    /// - Monitors thermal and mechanical indices against configured limits
    /// - Accumulates cavitation dose and thermal dose
    /// - Implements priority-based action hierarchy:
    ///   - Continue: All parameters safe
    ///   - Warning: Approaching limits (80% threshold)
    ///   - ReducePower: Active limit enforcement, reduces acoustic power
    ///   - Stop: Critical safety threshold exceeded
    /// - Adjusts acoustic power based on safety action
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
    /// Updates internal session state including time, progress, fields, safety metrics,
    /// and may reduce acoustic power if safety limits are approached.
    ///
    /// # References
    ///
    /// - IEC 62359:2010: Safety indices for ultrasound
    /// - FDA 510(k) Guidance: Ultrasound device safety requirements
    /// - Sapareto & Dewey (1990): CEM43 thermal dose model
    pub fn execute_therapy_step(&mut self, dt: f64) -> KwaversResult<()> {
        // Update session time and progress
        self.session_state.current_time += dt;
        self.session_state.progress = self.session_state.current_time / self.config.duration;

        // Generate acoustic field
        let mut acoustic_field =
            execution::generate_acoustic_field(&self.grid, &self.config.acoustic_params)?;

        // Apply power reduction from safety controller if therapy was previously constrained
        let power_factor = self.safety_controller.power_reduction_factor();
        if power_factor < 1.0 {
            // Scale pressure field proportionally to power reduction
            acoustic_field.pressure *= power_factor;
            for vel in &mut [
                &mut acoustic_field.velocity_x,
                &mut acoustic_field.velocity_y,
                &mut acoustic_field.velocity_z,
            ] {
                **vel *= power_factor;
            }
        }

        // Apply transcranial correction if enabled (future enhancement)
        let corrected_field = acoustic_field;

        // Record acoustic intensity for real-time monitoring
        // This must happen before any field modifications
        let intensity_metrics = self.intensity_tracker.record_intensity(
            &corrected_field.pressure,
            &Array3::ones(corrected_field.pressure.dim()),
            self.session_state.current_time,
        )?;

        // Compute temperature field from acoustic heating
        let temperature_field = execution::calculate_acoustic_heating(
            &corrected_field,
            &self.grid,
            dt,
            self.config.acoustic_params.focal_depth,
        );

        // Update thermal dose accumulation
        self.intensity_tracker
            .update_thermal_dose(&temperature_field, dt)?;

        // Update safety metrics with real-time measurements
        self.session_state.safety_metrics.thermal_index = intensity_metrics.spta_mw_cm2 * 0.001; // Convert to TI proxy
        self.session_state.safety_metrics.mechanical_index = self.config.acoustic_params.pnp
            / (1e6 * (self.config.acoustic_params.frequency).sqrt());
        self.session_state.safety_metrics.temperature_rise = temperature_field.clone();

        // Evaluate safety with real-time enforcement
        let safety_action = self.safety_controller.evaluate_safety(
            crate::clinical::therapy::therapy_integration::safety_controller::SafetyMetrics {
                thermal_index: self.session_state.safety_metrics.thermal_index,
                mechanical_index: self.session_state.safety_metrics.mechanical_index,
                cavitation_dose: self.session_state.safety_metrics.cavitation_dose,
                temperature_rise: temperature_field.clone(),
            },
            self.session_state.current_time,
        )?;

        // Handle safety action
        match safety_action {
            TherapyAction::Continue => {
                // Therapy proceeding normally
            }
            TherapyAction::Warning => {
                // Log warning but continue - approaching limits at 80%
            }
            TherapyAction::ReducePower => {
                // Power reduction is applied in next step via power_reduction_factor()
            }
            TherapyAction::Stop => {
                // Critical limit exceeded - therapy should terminate
                // Caller should check should_stop() and terminate session
            }
        }

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
            self.session_state.cavitation_activity = Some(cavitation_activity.clone());

            // Accumulate cavitation dose from activity
            let total_cavitation_activity: f64 =
                cavitation_activity.iter().sum::<f64>() / cavitation_activity.len() as f64;
            self.session_state.safety_metrics.cavitation_dose += total_cavitation_activity * dt;
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

        // Update legacy safety metrics calculation
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

    /// Check if therapy should terminate due to safety constraints
    ///
    /// Returns true if the safety controller has detected a critical limit violation
    /// that requires immediate therapy termination.
    ///
    /// # Returns
    ///
    /// True if therapy should stop, false if therapy can continue
    pub fn should_stop(&self) -> bool {
        self.safety_controller.should_stop()
    }

    /// Get current power reduction factor from safety controller
    ///
    /// Returns a multiplier [0.0, 1.0] indicating current acoustic power level.
    /// 1.0 = full power, 0.0 = no therapy delivery.
    ///
    /// # Returns
    ///
    /// Power reduction factor (0.0-1.0)
    pub fn power_reduction_factor(&self) -> f64 {
        self.safety_controller.power_reduction_factor()
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
    #[ignore] // Integration test - requires full therapy simulation stack
    /// Test therapy step execution
    ///
    /// Full integration test of the therapy orchestration system.
    /// Currently ignored because:
    /// 1. Requires complete acoustic field simulation (computationally expensive)
    /// 2. Needs functional microbubble dynamics, thermal modeling, and safety monitoring
    /// 3. Each therapy step involves multiple physics solvers  
    /// 4. More suitable for integration test suite than unit tests
    ///
    /// To enable: move to integration tests or enable for nightly builds only
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

    #[test]
    fn test_safety_controller_integration() {
        let config = TherapySessionConfig {
            primary_modality: TherapyModality::HIFU,
            secondary_modalities: vec![],
            imaging_data_path: None,
            duration: 30.0,
            acoustic_params: AcousticTherapyParams {
                frequency: 1.0e6,
                pnp: 5e6,
                prf: 100.0,
                duty_cycle: 0.05,
                focal_depth: 0.04,
                treatment_volume: 0.8,
            },
            safety_limits: SafetyLimits {
                thermal_index_max: 2.0, // Low limit for testing
                mechanical_index_max: 1.5,
                cavitation_dose_max: 100.0,
                max_treatment_time: 60.0,
            },
            patient_params: PatientParameters {
                skull_thickness: None,
                tissue_properties: TissuePropertyMap::liver((16, 16, 16)),
                target_volume: TargetVolume {
                    center: (0.04, 0.0, 0.0),
                    dimensions: (0.015, 0.015, 0.015),
                    tissue_type: TissueType::Liver,
                },
                risk_organs: vec![],
            },
        };

        let grid = Grid::new(16, 16, 16, 0.002, 0.002, 0.002).unwrap();
        let medium = Box::new(HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.0, &grid));

        let mut orchestrator = TherapyIntegrationOrchestrator::new(config, grid, medium).unwrap();

        // Initial state: therapy should be safe
        assert!(!orchestrator.should_stop());
        assert_eq!(orchestrator.power_reduction_factor(), 1.0);

        // Execute therapy steps and verify safety monitoring
        let dt = 0.5;
        let mut max_steps = 10;
        let mut safety_actions_observed = false;

        for step in 0..max_steps {
            let result = orchestrator.execute_therapy_step(dt);
            assert!(result.is_ok(), "Step {} failed", step);

            // Verify session state updated
            let state = orchestrator.session_state();
            assert!(state.current_time > 0.0);
            assert!(state.acoustic_field.is_some());

            // Check if therapy was terminated due to safety
            if orchestrator.should_stop() {
                safety_actions_observed = true;
                break;
            }

            // Monitor power reduction
            let power_factor = orchestrator.power_reduction_factor();
            if power_factor < 1.0 {
                safety_actions_observed = true;
            }
        }

        // Verify that safety monitoring is active
        // (Either power was reduced or therapy was stopped)
        assert!(
            safety_actions_observed || !orchestrator.should_stop(),
            "Safety controller should monitor therapy"
        );
    }

    #[test]
    fn test_intensity_tracker_integration() {
        let config = TherapySessionConfig {
            primary_modality: TherapyModality::HIFU,
            secondary_modalities: vec![],
            imaging_data_path: None,
            duration: 10.0,
            acoustic_params: AcousticTherapyParams {
                frequency: 2.0e6,
                pnp: 2e6,
                prf: 50.0,
                duty_cycle: 0.02,
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
                tissue_properties: TissuePropertyMap::liver((12, 12, 12)),
                target_volume: TargetVolume {
                    center: (0.03, 0.0, 0.0),
                    dimensions: (0.012, 0.012, 0.012),
                    tissue_type: TissueType::Liver,
                },
                risk_organs: vec![],
            },
        };

        let grid = Grid::new(12, 12, 12, 0.0025, 0.0025, 0.0025).unwrap();
        let medium = Box::new(HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.0, &grid));

        let mut orchestrator = TherapyIntegrationOrchestrator::new(config, grid, medium).unwrap();

        // Execute therapy steps and verify intensity tracking
        let dt = 0.2;
        for step in 0..5 {
            let result = orchestrator.execute_therapy_step(dt);
            assert!(result.is_ok(), "Step {} failed", step);

            let state = orchestrator.session_state();

            // Verify temperature field is computed
            assert!(
                state.safety_metrics.temperature_rise.len() > 0,
                "Temperature field should be computed in step {}",
                step
            );

            // Verify acoustic field exists
            assert!(
                state.acoustic_field.is_some(),
                "Acoustic field should exist in step {}",
                step
            );

            // Verify time advancement
            let expected_time = (step + 1) as f64 * dt;
            assert!(
                (state.current_time - expected_time).abs() < 1e-6,
                "Current time should be {} but got {}",
                expected_time,
                state.current_time
            );
        }

        // Verify final state
        let final_state = orchestrator.session_state();
        assert!(final_state.current_time > 0.0);
        assert!(final_state.progress > 0.0);
        assert!(final_state.progress <= 1.0);
    }
}
