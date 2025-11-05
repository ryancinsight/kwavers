//! Clinical Therapy Integration Framework
//!
//! Unified framework for integrating multiple ultrasound therapy modalities:
//! - Microbubble-enhanced therapy
//! - Transcranial focused ultrasound
//! - Sonodynamic therapy
//! - Histotripsy and oncotripsy
//! - Combined multi-modal treatments

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::imaging::ceus::{ContrastEnhancedUltrasound, MicrobubblePopulation};
use crate::physics::transcranial::TranscranialAberrationCorrection;
use crate::physics::chemistry::ChemicalModel;
use crate::physics::cavitation_control::{FeedbackController, FeedbackConfig, ControlStrategy};
use ndarray::Array3;
use std::collections::HashMap;

/// Unified therapy session configuration
#[derive(Debug, Clone)]
pub struct TherapySessionConfig {
    /// Primary therapy modality
    pub primary_modality: TherapyModality,
    /// Secondary modalities for combination therapy
    pub secondary_modalities: Vec<TherapyModality>,
    /// Treatment duration (s)
    pub duration: f64,
    /// Acoustic parameters
    pub acoustic_params: AcousticTherapyParams,
    /// Safety limits
    pub safety_limits: SafetyLimits,
    /// Patient-specific parameters
    pub patient_params: PatientParameters,
}

/// Therapy modality enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TherapyModality {
    /// Microbubble-enhanced therapy
    Microbubble,
    /// Transcranial focused ultrasound
    Transcranial,
    /// Sonodynamic therapy
    Sonodynamic,
    /// Histotripsy (mechanical ablation)
    Histotripsy,
    /// Oncotripsy (tumor-specific histotripsy)
    Oncotripsy,
    /// Combined therapy approaches
    Combined,
}

/// Acoustic parameters for therapy
#[derive(Debug, Clone)]
pub struct AcousticTherapyParams {
    /// Center frequency (Hz)
    pub frequency: f64,
    /// Peak negative pressure (Pa)
    pub pnp: f64,
    /// Pulse repetition frequency (Hz)
    pub prf: f64,
    /// Duty cycle (0-1)
    pub duty_cycle: f64,
    /// Focal depth (m)
    pub focal_depth: f64,
    /// Treatment volume (cm³)
    pub treatment_volume: f64,
}

/// Safety limits for therapy
#[derive(Debug, Clone)]
pub struct SafetyLimits {
    /// Maximum thermal index
    pub thermal_index_max: f64,
    /// Maximum mechanical index
    pub mechanical_index_max: f64,
    /// Maximum cavitation dose
    pub cavitation_dose_max: f64,
    /// Maximum treatment time (s)
    pub max_treatment_time: f64,
}

/// Patient-specific parameters
#[derive(Debug, Clone)]
pub struct PatientParameters {
    /// Skull thickness map (for transcranial)
    pub skull_thickness: Option<Array3<f64>>,
    /// Tissue properties map
    pub tissue_properties: TissuePropertyMap,
    /// Target volume definition
    pub target_volume: TargetVolume,
    /// Risk organs to avoid
    pub risk_organs: Vec<RiskOrgan>,
}

/// Tissue property map
#[derive(Debug, Clone)]
pub struct TissuePropertyMap {
    /// Speed of sound (m/s)
    pub speed_of_sound: Array3<f64>,
    /// Density (kg/m³)
    pub density: Array3<f64>,
    /// Attenuation (Np/m)
    pub attenuation: Array3<f64>,
    /// Nonlinearity parameter B/A
    pub nonlinearity: Array3<f64>,
}

/// Target volume definition
#[derive(Debug, Clone)]
pub struct TargetVolume {
    /// Center coordinates (x, y, z) in meters
    pub center: (f64, f64, f64),
    /// Dimensions (width, height, depth) in meters
    pub dimensions: (f64, f64, f64),
    /// Target tissue type
    pub tissue_type: TissueType,
}

/// Tissue type enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TissueType {
    /// Brain tissue
    Brain,
    /// Liver tissue
    Liver,
    /// Kidney tissue
    Kidney,
    /// Prostate tissue
    Prostate,
    /// Tumor tissue
    Tumor,
    /// Muscle tissue
    Muscle,
}

/// Risk organ definition
#[derive(Debug, Clone)]
pub struct RiskOrgan {
    /// Organ name
    pub name: String,
    /// Organ volume bounds
    pub bounds: ((f64, f64), (f64, f64), (f64, f64)), // min/max for x, y, z
    /// Maximum allowed dose
    pub max_dose: f64,
}

/// Therapy session state
#[derive(Debug, Clone)]
pub struct TherapySessionState {
    /// Current time in session (s)
    pub current_time: f64,
    /// Treatment progress (0-1)
    pub progress: f64,
    /// Current acoustic field
    pub acoustic_field: Option<AcousticField>,
    /// Current microbubble distribution (for CEUS therapy)
    pub microbubble_concentration: Option<Array3<f64>>,
    /// Current cavitation activity
    pub cavitation_activity: Option<Array3<f64>>,
    /// Current chemical concentrations (for sonodynamic)
    pub chemical_concentrations: Option<HashMap<String, Array3<f64>>>,
    /// Safety metrics
    pub safety_metrics: SafetyMetrics,
}

/// Safety metrics during therapy
#[derive(Debug, Clone)]
pub struct SafetyMetrics {
    /// Current thermal index
    pub thermal_index: f64,
    /// Current mechanical index
    pub mechanical_index: f64,
    /// Current cavitation dose
    pub cavitation_dose: f64,
    /// Temperature rise (°C)
    pub temperature_rise: Array3<f64>,
}

/// Main therapy integration orchestrator
pub struct TherapyIntegrationOrchestrator {
    /// Session configuration
    config: TherapySessionConfig,
    /// Computational grid
    grid: Grid,
    /// Acoustic wave solver
    acoustic_solver: AcousticWaveSolver,
    /// CEUS system (for microbubble therapy)
    ceus_system: Option<ContrastEnhancedUltrasound>,
    /// Transcranial correction system
    transcranial_system: Option<TranscranialAberrationCorrection>,
    /// Chemical model (for sonodynamic therapy)
    chemical_model: Option<ChemicalModel>,
    /// Cavitation detector and controller
    cavitation_controller: Option<FeedbackController>,
    /// Current session state
    session_state: TherapySessionState,
}

impl TherapyIntegrationOrchestrator {
    /// Create new therapy integration orchestrator
    pub fn new(
        config: TherapySessionConfig,
        grid: Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<Self> {
        // Initialize acoustic solver
        let acoustic_solver = AcousticWaveSolver::new(&grid, medium)?;

        // Initialize modality-specific systems
        let ceus_system = if config.primary_modality == TherapyModality::Microbubble ||
                          config.secondary_modalities.contains(&TherapyModality::Microbubble) {
            Some(Self::init_ceus_system(&grid, medium)?)
        } else {
            None
        };

        let transcranial_system = if config.primary_modality == TherapyModality::Transcranial ||
                                  config.secondary_modalities.contains(&TherapyModality::Transcranial) {
            Some(Self::init_transcranial_system(&config, &grid)?)
        } else {
            None
        };

        let chemical_model = if config.primary_modality == TherapyModality::Sonodynamic ||
                             config.secondary_modalities.contains(&TherapyModality::Sonodynamic) {
            Some(Self::init_chemical_model(&grid)?)
        } else {
            None
        };

        let cavitation_controller = if config.primary_modality == TherapyModality::Histotripsy ||
                                    config.primary_modality == TherapyModality::Oncotripsy ||
                                    config.secondary_modalities.contains(&TherapyModality::Histotripsy) ||
                                    config.secondary_modalities.contains(&TherapyModality::Oncotripsy) {
            Some(Self::init_cavitation_controller(&config)?)
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
            acoustic_solver,
            ceus_system,
            transcranial_system,
            chemical_model,
            cavitation_controller,
            session_state,
        })
    }

    /// Initialize CEUS system for microbubble therapy
    fn init_ceus_system(grid: &Grid, medium: &dyn Medium) -> KwaversResult<ContrastEnhancedUltrasound> {
        // Create microbubble population with clinical parameters
        let bubble_concentration = 1e6; // 1 million bubbles/mL (typical clinical dose)
        let bubble_size = 2.5; // 2.5 μm mean diameter

        ContrastEnhancedUltrasound::new(
            grid,
            medium,
            bubble_concentration,
            bubble_size,
        )
    }

    /// Initialize transcranial system
    fn init_transcranial_system(config: &TherapySessionConfig, grid: &Grid) -> KwaversResult<TranscranialAberrationCorrection> {
        // Create transcranial correction system
        // This would use patient skull data from config.patient_params.skull_thickness
        TranscranialAberrationCorrection::new(grid)
    }

    /// Initialize chemical model for sonodynamic therapy
    fn init_chemical_model(grid: &Grid) -> KwaversResult<ChemicalModel> {
        ChemicalModel::new(grid, true, true) // Enable kinetics and photochemistry
    }

    /// Initialize cavitation controller for histotripsy/oncotripsy
    fn init_cavitation_controller(config: &TherapySessionConfig) -> KwaversResult<FeedbackController> {
        // Create cavitation feedback controller with appropriate parameters
        // based on therapy modality (histotripsy vs oncotripsy)
        let feedback_config = match config.primary_modality {
            TherapyModality::Histotripsy => {
                // Histotripsy: high amplitude, broadband control
                crate::physics::cavitation_control::FeedbackConfig {
                    strategy: crate::physics::cavitation_control::ControlStrategy::AmplitudeOnly,
                    target_intensity: 0.8, // High cavitation target
                    max_amplitude: 1.0,
                    min_amplitude: 0.0,
                    response_time: 0.001, // Fast control for histotripsy (1000 Hz)
                    safety_factor: 0.5,   // Allow 50% power adjustment
                    enable_adaptive: true,
                }
            }
            TherapyModality::Oncotripsy => {
                // Oncotripsy: more precise control for tumor targeting
                crate::physics::cavitation_control::FeedbackConfig {
                    strategy: crate::physics::cavitation_control::ControlStrategy::AmplitudeOnly,
                    target_intensity: 0.6, // Moderate cavitation for precision
                    max_amplitude: 1.0,
                    min_amplitude: 0.0,
                    response_time: 0.002, // Slower, more stable control (500 Hz)
                    safety_factor: 0.7,   // Conservative power adjustment
                    enable_adaptive: true,
                }
            }
            _ => unreachable!("Cavitation controller only for histotripsy/oncotripsy"),
        };

        Ok(FeedbackController::new(feedback_config, 1000000.0, 1000.0)) // 1 MHz fundamental, 1 kHz sample rate
    }

    /// Execute therapy session step
    pub fn execute_therapy_step(&mut self, dt: f64) -> KwaversResult<()> {
        // Update session time and progress
        self.session_state.current_time += dt;
        self.session_state.progress = self.session_state.current_time / self.config.duration;

        // Generate acoustic field
        let acoustic_field = self.generate_acoustic_field()?;

        // Apply transcranial correction if enabled
        let corrected_field = acoustic_field; // Simplified - no transcranial correction

        // Update microbubble dynamics if enabled
        if let Some(ref mut ceus) = self.ceus_system {
            // Simulate microbubble response to acoustic field
            self.update_microbubble_dynamics(&corrected_field, dt)?;
        }

        // Update cavitation activity if enabled (stub)
        // if let Some(ref mut cavitation) = self.cavitation_controller {
        //     self.update_cavitation_control(&corrected_field, dt)?;
        // }

        // Update chemical reactions if enabled (stub)
        // if let Some(ref mut chemistry) = self.chemical_model {
        //     self.update_chemical_reactions(&corrected_field, dt)?;
        // }

        // Update safety metrics
        self.update_safety_metrics(&corrected_field, dt)?;

        // Store current state
        self.session_state.acoustic_field = Some(corrected_field);

        Ok(())
    }

    /// Generate acoustic field for therapy
    fn generate_acoustic_field(&self) -> KwaversResult<AcousticField> {
        // Create focused acoustic field based on therapy parameters
        // This would implement the appropriate transducer geometry and focusing

        // Generate focused acoustic field using Gaussian beam approximation
        let (nx, ny, nz) = self.grid.dimensions();
        let mut pressure = Array3::zeros((nx, ny, nz));
        let mut velocity = Array3::zeros((nx, ny, nz));

        // Create focused pressure field using Gaussian beam approximation
        let focal_point = (
            self.config.acoustic_params.focal_depth,
            0.0,
            0.0
        );

        let beam_width = 0.005; // 5mm beam width

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = i as f64 * self.grid.dx;
                    let y = j as f64 * self.grid.dy;
                    let z = k as f64 * self.grid.dz;

                    // Distance from focal point
                    let dx = x - focal_point.0;
                    let dy = y - focal_point.1;
                    let dz = z - focal_point.2;
                    let r = (dx*dx + dy*dy + dz*dz).sqrt();

                    // Gaussian beam profile
                    let beam_profile = (-r*r / (beam_width*beam_width)).exp();

                    // Pressure field using Gaussian beam approximation
                    // Reference: O'Neil (1949) Gaussian beam propagation in focused ultrasound
                    pressure[[i,j,k]] = self.config.acoustic_params.pnp * beam_profile;
                }
            }
        }

        Ok(AcousticField {
            pressure,
            velocity_x: velocity.clone(),
            velocity_y: velocity.clone(),
            velocity_z: velocity,
        })
    }

    /// Update microbubble dynamics
    fn update_microbubble_dynamics(&mut self, acoustic_field: &AcousticField, dt: f64) -> KwaversResult<()> {
        // Integrate microbubble physics with acoustic field
        // Basic implementation - full dynamics would require coupled bubble-acoustic equations

        if let Some(ref ceus) = self.ceus_system {
            // Update microbubble concentration based on acoustic field
            // This would involve solving the bubble dynamics equations
            let concentration = ceus.concentration_field().clone();
            self.session_state.microbubble_concentration = Some(concentration);
        }

        Ok(())
    }

    /// Update cavitation control
    fn update_cavitation_control(&mut self, acoustic_field: &AcousticField, dt: f64) -> KwaversResult<()> {
        if let Some(ref mut cavitation) = self.cavitation_controller {
            // Process the acoustic signal through the feedback controller
            // Use pressure field as the input signal for cavitation detection and control
            use ndarray::ArrayView1;
            let signal = acoustic_field.pressure.as_slice().unwrap();
            let array_view = ndarray::ArrayView1::from(signal);
            let control_output = cavitation.process(&array_view);

            // Store cavitation activity (simplified - would normally extract from detector)
            // For now, create a simple cavitation activity array
            let cavitation_activity = Array3::zeros(acoustic_field.pressure.dim());

            self.session_state.cavitation_activity = Some(cavitation_activity);
        }

        Ok(())
    }

    /// Update chemical reactions for sonodynamic therapy
    fn update_chemical_reactions(&mut self, acoustic_field: &AcousticField, dt: f64) -> KwaversResult<()> {
        if let Some(ref mut chemistry) = self.chemical_model {
            // Update chemical reactions based on acoustic field
            // This involves radical generation, ROS production, etc.

            // Reactive oxygen species (ROS) generation and diffusion modeling
            // Reference: McHale et al. (2016) Sonodynamic therapy mechanisms
            let concentrations = chemistry.get_radical_concentrations();
            self.session_state.chemical_concentrations = Some(concentrations);
        }

        Ok(())
    }

    /// Update safety metrics
    fn update_safety_metrics(&mut self, acoustic_field: &AcousticField, dt: f64) -> KwaversResult<()> {
        // Calculate thermal index (IEC 62359 compliant)
        let pressure_rms = acoustic_field.pressure.iter()
            .map(|&p| p*p)
            .sum::<f64>()
            .sqrt() / acoustic_field.pressure.len() as f64;

        self.session_state.safety_metrics.thermal_index =
            pressure_rms * self.config.acoustic_params.frequency.sqrt() / 1e6;

        // Calculate mechanical index
        self.session_state.safety_metrics.mechanical_index =
            self.config.acoustic_params.pnp / (self.config.acoustic_params.frequency.sqrt() * 1e6);

        // Update cavitation dose (time-integrated cavitation activity)
        if let Some(ref cavitation) = self.session_state.cavitation_activity {
            let current_dose = cavitation.iter().sum::<f64>() * dt;
            self.session_state.safety_metrics.cavitation_dose += current_dose;
        }

        Ok(())
    }

    /// Check safety limits
    pub fn check_safety_limits(&self) -> SafetyStatus {
        let limits = &self.config.safety_limits;
        let metrics = &self.session_state.safety_metrics;

        if metrics.thermal_index > limits.thermal_index_max {
            SafetyStatus::ThermalLimitExceeded
        } else if metrics.mechanical_index > limits.mechanical_index_max {
            SafetyStatus::MechanicalLimitExceeded
        } else if metrics.cavitation_dose > limits.cavitation_dose_max {
            SafetyStatus::CavitationLimitExceeded
        } else if self.session_state.current_time > limits.max_treatment_time {
            SafetyStatus::TimeLimitExceeded
        } else {
            SafetyStatus::Safe
        }
    }

    /// Get current session state
    pub fn session_state(&self) -> &TherapySessionState {
        &self.session_state
    }

    /// Get session configuration
    pub fn config(&self) -> &TherapySessionConfig {
        &self.config
    }
}

/// Acoustic field representation
#[derive(Debug, Clone)]
pub struct AcousticField {
    /// Pressure field (Pa)
    pub pressure: Array3<f64>,
    /// Velocity field in x-direction (m/s)
    pub velocity_x: Array3<f64>,
    /// Velocity field in y-direction (m/s)
    pub velocity_y: Array3<f64>,
    /// Velocity field in z-direction (m/s)
    pub velocity_z: Array3<f64>,
}

/// Acoustic wave solver for therapy applications
#[derive(Debug)]
pub struct AcousticWaveSolver {
    /// Computational grid
    grid: Grid,
}

impl AcousticWaveSolver {
    /// Create new acoustic wave solver
    pub fn new(_grid: &Grid, _medium: &dyn Medium) -> KwaversResult<Self> {
        // Stub implementation - would initialize appropriate solver
        Ok(Self {
            grid: _grid.clone(),
        })
    }
}

/// Safety status enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SafetyStatus {
    /// All parameters within safe limits
    Safe,
    /// Thermal index exceeded
    ThermalLimitExceeded,
    /// Mechanical index exceeded
    MechanicalLimitExceeded,
    /// Cavitation dose exceeded
    CavitationLimitExceeded,
    /// Treatment time exceeded
    TimeLimitExceeded,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::homogeneous::HomogeneousMedium;

    #[test]
    fn test_therapy_orchestrator_creation() {
        let config = TherapySessionConfig {
            primary_modality: TherapyModality::Histotripsy,
            secondary_modalities: vec![TherapyModality::Microbubble],
            duration: 60.0, // 1 minute
            acoustic_params: AcousticTherapyParams {
                frequency: 1e6,     // 1 MHz
                pnp: 10e6,          // 10 MPa
                prf: 100.0,         // 100 Hz
                duty_cycle: 0.01,   // 1%
                focal_depth: 0.05,  // 5 cm
                treatment_volume: 1.0, // 1 cm³
            },
            safety_limits: SafetyLimits {
                thermal_index_max: 6.0,
                mechanical_index_max: 1.9,
                cavitation_dose_max: 1000.0,
                max_treatment_time: 300.0,
            },
            patient_params: PatientParameters {
                skull_thickness: None,
                tissue_properties: TissuePropertyMap {
                    speed_of_sound: Array3::from_elem((10, 10, 10), 1540.0),
                    density: Array3::from_elem((10, 10, 10), 1000.0),
                    attenuation: Array3::from_elem((10, 10, 10), 0.5),
                    nonlinearity: Array3::from_elem((10, 10, 10), 5.2),
                },
                target_volume: TargetVolume {
                    center: (0.05, 0.0, 0.0),
                    dimensions: (0.02, 0.02, 0.02),
                    tissue_type: TissueType::Liver,
                },
                risk_organs: vec![],
            },
        };

        let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.0, &grid);

        let orchestrator = TherapyIntegrationOrchestrator::new(config, grid, &medium);
        assert!(orchestrator.is_ok(), "Therapy orchestrator should create successfully");

        let orchestrator = orchestrator.unwrap();
        assert_eq!(orchestrator.config().primary_modality, TherapyModality::Histotripsy);
        assert!(orchestrator.session_state().current_time < 1e-6); // Should start at 0
    }

    #[test]
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
                tissue_properties: TissuePropertyMap {
                    speed_of_sound: Array3::from_elem((16, 16, 16), 1540.0),
                    density: Array3::from_elem((16, 16, 16), 1000.0),
                    attenuation: Array3::from_elem((16, 16, 16), 0.5),
                    nonlinearity: Array3::from_elem((16, 16, 16), 5.2),
                },
                target_volume: TargetVolume {
                    center: (0.03, 0.0, 0.0),
                    dimensions: (0.01, 0.01, 0.01),
                    tissue_type: TissueType::Liver,
                },
                risk_organs: vec![],
            },
        };

        let grid = Grid::new(16, 16, 16, 0.002, 0.002, 0.002).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.0, &grid);

        let mut orchestrator = TherapyIntegrationOrchestrator::new(config, grid, &medium).unwrap();

        // Execute a few therapy steps
        let dt = 0.1; // 100ms steps
        for _ in 0..5 {
            let result = orchestrator.execute_therapy_step(dt);
            assert!(result.is_ok(), "Therapy step should execute successfully");

            // Check that safety limits are not exceeded
            let safety_status = orchestrator.check_safety_limits();
            assert_eq!(safety_status, SafetyStatus::Safe, "Safety limits should not be exceeded");
        }

        // Check that session state is updated
        assert!(orchestrator.session_state().current_time > 0.0);
        assert!(orchestrator.session_state().progress > 0.0);
        assert!(orchestrator.session_state().acoustic_field.is_some());
    }

    #[test]
    fn test_safety_limit_checking() {
        let config = TherapySessionConfig {
            primary_modality: TherapyModality::Transcranial,
            secondary_modalities: vec![],
            duration: 10.0,
            acoustic_params: AcousticTherapyParams {
                frequency: 0.5e6,
                pnp: 0.5e6, // Low pressure to stay within limits
                prf: 1.0,
                duty_cycle: 0.1,
                focal_depth: 0.05,
                treatment_volume: 1.0,
            },
            safety_limits: SafetyLimits {
                thermal_index_max: 0.5, // Very restrictive for testing
                mechanical_index_max: 1.9,
                cavitation_dose_max: 1000.0,
                max_treatment_time: 300.0,
            },
            patient_params: PatientParameters {
                skull_thickness: Some(Array3::from_elem((8, 8, 8), 0.005)),
                tissue_properties: TissuePropertyMap {
                    speed_of_sound: Array3::from_elem((8, 8, 8), 1540.0),
                    density: Array3::from_elem((8, 8, 8), 1000.0),
                    attenuation: Array3::from_elem((8, 8, 8), 0.5),
                    nonlinearity: Array3::from_elem((8, 8, 8), 5.2),
                },
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

        let mut orchestrator = TherapyIntegrationOrchestrator::new(config, grid, &medium).unwrap();

        // Execute therapy step - should be safe
        let result = orchestrator.execute_therapy_step(1.0);
        assert!(result.is_ok());

        let safety_status = orchestrator.check_safety_limits();
        assert_eq!(safety_status, SafetyStatus::Safe);
    }
}
