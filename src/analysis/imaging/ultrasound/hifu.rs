//! HIFU (High-Intensity Focused Ultrasound) treatment planning types

/// HIFU transducer configuration
#[derive(Debug, Clone)]
pub struct HIFUTransducer {
    /// Transducer frequency (Hz)
    pub frequency: f64,
    /// Transducer power (W)
    pub power: f64,
    /// Focal length (mm)
    pub focal_length: f64,
    /// Aperture diameter (mm)
    pub aperture: f64,
}

impl Default for HIFUTransducer {
    fn default() -> Self {
        Self {
            frequency: 1.0e6,
            power: 100.0,
            focal_length: 12.0,
            aperture: 64.0,
        }
    }
}

/// Target shape for HIFU treatment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetShape {
    /// Spherical target
    Sphere,
    /// Ellipsoidal target
    Ellipsoid,
    /// Cylindrical target
    Cylinder,
}

/// Treatment target specification
#[derive(Debug, Clone)]
pub struct TreatmentTarget {
    /// Target center position (mm)
    pub center: (f64, f64, f64),
    /// Target shape
    pub shape: TargetShape,
    /// Target dimensions (mm)
    pub dimensions: (f64, f64, f64),
    /// Target temperature threshold (°C)
    pub temperature_threshold: f64,
}

/// Treatment protocol parameters
#[derive(Debug, Clone)]
pub struct TreatmentProtocol {
    /// Sonication duration (s)
    pub duration: f64,
    /// Duty cycle (0.0-1.0)
    pub duty_cycle: f64,
    /// Treatment temperature (°C)
    pub target_temperature: f64,
}

/// HIFU treatment plan
#[derive(Debug, Clone)]
pub struct HIFUTreatmentPlan {
    /// Treatment transducer
    pub transducer: HIFUTransducer,
    /// Treatment target
    pub target: TreatmentTarget,
    /// Treatment protocol
    pub protocol: TreatmentProtocol,
}
