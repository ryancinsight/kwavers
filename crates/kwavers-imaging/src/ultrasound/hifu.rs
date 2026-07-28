//! HIFU domain definitions.

use aequitas::systems::si::quantities::{
    Frequency, Intensity, Length, Power, TemperatureDifference, ThermodynamicTemperature, Time,
};

/// HIFU transducer configuration
#[derive(Debug, Clone)]
pub struct DomainHIFUTransducer {
    /// Transducer geometry
    pub geometry: HifuTransducerGeometry,
    /// Operating frequency in the SI base unit hertz.
    pub frequency: Frequency<f64>,
    /// Acoustic power in the SI base unit watt.
    pub acoustic_power: Power<f64>,
    /// Focal length in the SI base unit metre.
    pub focal_length: Length<f64>,
    /// Aperture radius in the SI base unit metre.
    pub aperture_radius: Length<f64>,
    /// Duty cycle (0-1)
    pub duty_cycle: f64,
}

impl DomainHIFUTransducer {
    /// Create a new single-element focused transducer
    #[must_use]
    pub fn new_single_element(
        frequency: Frequency<f64>,
        acoustic_power: Power<f64>,
        focal_length: Length<f64>,
        aperture_radius: Length<f64>,
    ) -> Self {
        Self {
            geometry: HifuTransducerGeometry::SingleElement,
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
pub enum HifuTransducerGeometry {
    /// Single-element focused transducer
    SingleElement,
    /// Phased array transducer
    PhasedArray {
        /// Number of elements
        n_elements: usize,
        /// Element spacing in the SI base unit metre.
        element_spacing: Length<f64>,
    },
    /// Annular array transducer
    AnnularArray {
        /// Number of rings
        n_rings: usize,
        /// Ring radii in the SI base unit metre.
        ring_radii: Vec<Length<f64>>,
    },
}

/// Treatment planning and execution
#[derive(Debug, Clone)]
pub struct DomainHIFUTreatmentPlan {
    /// Target region definition
    pub target: TreatmentTarget,
    /// Treatment protocol
    pub protocol: HifuTreatmentProtocol,
    /// Safety margins and constraints
    pub safety: HifuSafetyConstraints,
    /// Monitoring configuration
    pub monitoring: HifuMonitoringConfig,
}

impl DomainHIFUTreatmentPlan {
    /// Create a new treatment plan
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(target: TreatmentTarget, protocol: HifuTreatmentProtocol) -> Self {
        Self {
            target,
            protocol,
            safety: HifuSafetyConstraints::default(),
            monitoring: HifuMonitoringConfig::default(),
        }
    }

    /// Validate treatment plan against safety constraints
    /// # Errors
    /// - Returns `KwaversError::Validation` if a validation-class constraint is violated.
    ///
    pub fn validate(
        &self,
        transducer: &DomainHIFUTransducer,
    ) -> Result<(), kwavers_core::error::KwaversError> {
        use kwavers_core::error::{KwaversError, ValidationError};

        // Check target is within accessible region
        let target_depth = self.target.center[2].into_base();
        let focal_length = transducer.focal_length.into_base();
        if target_depth < focal_length * 0.5 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "target.center.z".to_owned(),
                value: target_depth,
                reason: "Target too close to transducer".to_owned(),
            }));
        }

        // Check thermal constraints
        const MAX_SAFE_TEMPERATURE_KELVIN: f64 = 373.15;
        let max_temperature = self.safety.max_temperature.into_base();
        if max_temperature > MAX_SAFE_TEMPERATURE_KELVIN {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "safety.max_temperature".to_owned(),
                value: max_temperature,
                reason: "Maximum temperature exceeds safe limit".to_owned(),
            }));
        }

        // Check acoustic intensity limits
        const MAX_SAFE_INTENSITY_W_PER_M2: f64 = 1.0e7;
        let max_intensity = self.safety.max_intensity.into_base();
        if max_intensity > MAX_SAFE_INTENSITY_W_PER_M2 {
            // The public contract stores the legacy 1000 W/cm² limit as its SI
            // base-unit equivalent, 10^7 W/m².
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "safety.max_intensity".to_owned(),
                value: max_intensity,
                reason: "Maximum intensity exceeds safe limit".to_owned(),
            }));
        }

        Ok(())
    }
}

/// Treatment target specification
#[derive(Debug, Clone)]
pub struct TreatmentTarget {
    /// Target center position in SI base-unit metres.
    pub center: [Length<f64>; 3],
    /// Target dimensions in SI base-unit metres.
    pub dimensions: [Length<f64>; 3],
    /// Target shape
    pub shape: HifuTargetShape,
}

/// Target shape types
#[derive(Debug, Clone, PartialEq)]
pub enum HifuTargetShape {
    /// Spherical target
    Sphere,
    /// Cylindrical target
    Cylinder,
    /// Custom shape defined by mask
    Custom,
}

/// Treatment protocol parameters
#[derive(Debug, Clone)]
pub struct HifuTreatmentProtocol {
    /// Total treatment time in the SI base unit second.
    pub total_duration: Time<f64>,
    /// Pulse duration in the SI base unit second.
    pub pulse_duration: Time<f64>,
    /// Pulse repetition frequency in the SI base unit hertz.
    pub prf: Frequency<f64>,
    /// Cooling periods between pulses in the SI base unit second.
    pub cooling_period: Time<f64>,
    /// Treatment phases
    pub phases: Vec<TreatmentPhase>,
}

/// Treatment phase definition
#[derive(Debug, Clone)]
pub struct TreatmentPhase {
    /// Phase name
    pub name: String,
    /// Phase duration in the SI base unit second.
    pub duration: Time<f64>,
    /// Acoustic power during phase in the SI base unit watt.
    pub power: Power<f64>,
    /// Focus position offset from target center in SI base-unit metres.
    pub focus_offset: [Length<f64>; 3],
}

/// Safety constraints
#[derive(Debug, Clone)]
pub struct HifuSafetyConstraints {
    /// Maximum absolute temperature in SI base-unit kelvin.
    pub max_temperature: ThermodynamicTemperature<f64>,
    /// Maximum thermal dose (CEM43)
    pub max_thermal_dose: f64,
    /// Maximum acoustic intensity in the SI base unit watt per square metre.
    pub max_intensity: Intensity<f64>,
    /// Critical structure avoidance zones
    pub avoidance_zones: Vec<AvoidanceZone>,
}

impl Default for HifuSafetyConstraints {
    fn default() -> Self {
        Self {
            max_temperature: ThermodynamicTemperature::from_base(358.15), // 85 °C
            max_thermal_dose: 240.0,                                      // CEM43
            max_intensity: Intensity::from_base(1.0e7),                   // 1000 W/cm²
            avoidance_zones: Vec::new(),
        }
    }
}

/// Avoidance zone for critical structures
#[derive(Debug, Clone)]
pub struct AvoidanceZone {
    /// Zone center in SI base-unit metres.
    pub center: [Length<f64>; 3],
    /// Zone radius in the SI base unit metre.
    pub radius: Length<f64>,
    /// Maximum allowed temperature rise in the SI base unit kelvin.
    pub max_temp_rise: TemperatureDifference<f64>,
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct HifuMonitoringConfig {
    /// Temperature monitoring points
    pub temperature_points: Vec<[Length<f64>; 3]>,
    /// Acoustic feedback channels
    pub feedback_channels: Vec<FeedbackChannel>,
    /// Real-time adjustment parameters
    pub real_time_adjustment: bool,
}

impl Default for HifuMonitoringConfig {
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
