//! HIFU domain definitions

/// HIFU transducer configuration
#[derive(Debug, Clone)]
pub struct HIFUTransducer {
    /// Transducer geometry
    pub geometry: TransducerGeometry,
    /// Operating frequency (Hz)
    pub frequency: f64,
    /// Acoustic power (W)
    pub acoustic_power: f64,
    /// Focal length (m)
    pub focal_length: f64,
    /// Aperture radius (m)
    pub aperture_radius: f64,
    /// Duty cycle (0-1)
    pub duty_cycle: f64,
}

impl HIFUTransducer {
    /// Create a new single-element focused transducer
    pub fn new_single_element(
        frequency: f64,
        acoustic_power: f64,
        focal_length: f64,
        aperture_radius: f64,
    ) -> Self {
        Self {
            geometry: TransducerGeometry::SingleElement,
            frequency,
            acoustic_power,
            focal_length,
            aperture_radius,
            duty_cycle: 1.0,
        }
    }
}

/// Transducer geometry types
#[derive(Debug, Clone, PartialEq)]
pub enum TransducerGeometry {
    /// Single-element focused transducer
    SingleElement,
    /// Phased array transducer
    PhasedArray {
        /// Number of elements
        n_elements: usize,
        /// Element spacing (m)
        element_spacing: f64,
    },
    /// Annular array transducer
    AnnularArray {
        /// Number of rings
        n_rings: usize,
        /// Ring radii (m)
        ring_radii: Vec<f64>,
    },
}

/// Treatment planning and execution
#[derive(Debug, Clone)]
pub struct HIFUTreatmentPlan {
    /// Target region definition
    pub target: TreatmentTarget,
    /// Treatment protocol
    pub protocol: TreatmentProtocol,
    /// Safety margins and constraints
    pub safety: SafetyConstraints,
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
}

impl HIFUTreatmentPlan {
    /// Create a new treatment plan
    pub fn new(target: TreatmentTarget, protocol: TreatmentProtocol) -> Self {
        Self {
            target,
            protocol,
            safety: SafetyConstraints::default(),
            monitoring: MonitoringConfig::default(),
        }
    }

    /// Validate treatment plan against safety constraints
    pub fn validate(
        &self,
        transducer: &HIFUTransducer,
    ) -> Result<(), crate::domain::core::error::KwaversError> {
        use crate::core::error::{KwaversError, ValidationError};

        // Check target is within accessible region
        if self.target.center[2] < transducer.focal_length * 0.5 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "target.center.z".to_string(),
                value: self.target.center[2],
                reason: "Target too close to transducer".to_string(),
            }));
        }

        // Check thermal constraints
        if self.safety.max_temperature > 100.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "safety.max_temperature".to_string(),
                value: self.safety.max_temperature,
                reason: "Maximum temperature exceeds safe limit".to_string(),
            }));
        }

        // Check acoustic intensity limits
        if self.safety.max_intensity > 1000.0 {
            // 1000 W/cm² = 10^7 W/m²
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "safety.max_intensity".to_string(),
                value: self.safety.max_intensity,
                reason: "Maximum intensity exceeds safe limit".to_string(),
            }));
        }

        Ok(())
    }
}

/// Treatment target specification
#[derive(Debug, Clone)]
pub struct TreatmentTarget {
    /// Target center position (m)
    pub center: [f64; 3],
    /// Target dimensions (m)
    pub dimensions: [f64; 3],
    /// Target shape
    pub shape: TargetShape,
}

/// Target shape types
#[derive(Debug, Clone, PartialEq)]
pub enum TargetShape {
    /// Spherical target
    Sphere,
    /// Cylindrical target
    Cylinder,
    /// Custom shape defined by mask
    Custom,
}

/// Treatment protocol parameters
#[derive(Debug, Clone)]
pub struct TreatmentProtocol {
    /// Total treatment time (s)
    pub total_duration: f64,
    /// Pulse duration (s)
    pub pulse_duration: f64,
    /// Pulse repetition frequency (Hz)
    pub prf: f64,
    /// Cooling periods between pulses (s)
    pub cooling_period: f64,
    /// Treatment phases
    pub phases: Vec<TreatmentPhase>,
}

/// Treatment phase definition
#[derive(Debug, Clone)]
pub struct TreatmentPhase {
    /// Phase name
    pub name: String,
    /// Phase duration (s)
    pub duration: f64,
    /// Acoustic power during phase (W)
    pub power: f64,
    /// Focus position offset from target center (m)
    pub focus_offset: [f64; 3],
}

/// Safety constraints
#[derive(Debug, Clone)]
pub struct SafetyConstraints {
    /// Maximum temperature (°C)
    pub max_temperature: f64,
    /// Maximum thermal dose (CEM43)
    pub max_thermal_dose: f64,
    /// Maximum acoustic intensity (W/cm²)
    pub max_intensity: f64,
    /// Critical structure avoidance zones
    pub avoidance_zones: Vec<AvoidanceZone>,
}

impl Default for SafetyConstraints {
    fn default() -> Self {
        Self {
            max_temperature: 85.0,   // °C
            max_thermal_dose: 240.0, // CEM43
            max_intensity: 1000.0,   // W/cm²
            avoidance_zones: Vec::new(),
        }
    }
}

/// Avoidance zone for critical structures
#[derive(Debug, Clone)]
pub struct AvoidanceZone {
    /// Zone center (m)
    pub center: [f64; 3],
    /// Zone radius (m)
    pub radius: f64,
    /// Maximum allowed temperature rise (°C)
    pub max_temp_rise: f64,
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Temperature monitoring points
    pub temperature_points: Vec<[f64; 3]>,
    /// Acoustic feedback channels
    pub feedback_channels: Vec<FeedbackChannel>,
    /// Real-time adjustment parameters
    pub real_time_adjustment: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            temperature_points: Vec::new(),
            feedback_channels: vec![FeedbackChannel::Ultrasound],
            real_time_adjustment: true,
        }
    }
}

/// Feedback channel types
#[derive(Debug, Clone, PartialEq)]
pub enum FeedbackChannel {
    /// Magnetic Resonance Imaging
    MRI,
    /// Ultrasound imaging
    Ultrasound,
    /// Thermocouple
    Thermocouple,
    /// Infrared thermography
    Infrared,
}
