//! Core data structures for transcranial treatment planning

use ndarray::Array3;

/// Treatment target volume in brain coordinates
#[derive(Debug, Clone)]
pub struct TargetVolume {
    /// Target center (x, y, z) in mm relative to brain origin
    pub center: [f64; 3],
    /// Target dimensions (width, height, depth) in mm
    pub dimensions: [f64; 3],
    /// Target shape (ellipsoidal, rectangular, etc.)
    pub shape: TargetShape,
    /// Clinical priority (1-10, higher = more critical)
    pub priority: u8,
    /// Maximum allowable temperature (°C)
    pub max_temperature: f64,
    /// Required acoustic intensity (W/cm²)
    pub required_intensity: f64,
}

/// Permissible shapes for treatment targets
#[derive(Debug, Clone)]
pub enum TargetShape {
    /// Ellipsoidal target
    Ellipsoidal,
    /// Rectangular target
    Rectangular,
    /// Custom shape defined by mask
    Custom(Array3<bool>),
}

/// Complete treatment plan for tFUS session
#[derive(Debug)]
pub struct TreatmentPlan {
    /// Patient identifier
    pub patient_id: String,
    /// Treatment targets
    pub targets: Vec<TargetVolume>,
    /// Skull CT data (Hounsfield units)
    pub skull_ct: Array3<f64>,
    /// Optimal transducer positions and phases
    pub transducer_setup: TransducerSetup,
    /// Predicted acoustic field in brain
    pub acoustic_field: Array3<f64>,
    /// Predicted temperature field
    pub temperature_field: Array3<f64>,
    /// Safety margins and constraints
    pub safety_constraints: SafetyConstraints,
    /// Estimated treatment time (seconds)
    pub treatment_time: f64,
}

/// Transducer array configuration
#[derive(Debug, Clone)]
pub struct TransducerSetup {
    /// Number of transducer elements
    pub num_elements: usize,
    /// Element positions (x, y, z) in mm
    pub element_positions: Vec<[f64; 3]>,
    /// Element phases for aberration correction (radians)
    pub element_phases: Vec<f64>,
    /// Element amplitudes (normalized)
    pub element_amplitudes: Vec<f64>,
    /// Operating frequency (Hz)
    pub frequency: f64,
    /// Focal distance (mm)
    pub focal_distance: f64,
}

/// Safety constraints for treatment
#[derive(Debug, Clone)]
pub struct SafetyConstraints {
    /// Maximum skull surface temperature (°C)
    pub max_skull_temp: f64,
    /// Maximum brain temperature (°C)
    pub max_brain_temp: f64,
    /// Maximum mechanical index
    pub max_mi: f64,
    /// Maximum thermal dose (CEM43)
    pub max_thermal_dose: f64,
    /// Minimum distance from skull-brain interface (mm)
    pub min_skull_distance: f64,
}

impl Default for SafetyConstraints {
    fn default() -> Self {
        Self {
            max_skull_temp: 42.0,    // °C
            max_brain_temp: 43.0,    // °C
            max_mi: 1.9,             // Mechanical index limit
            max_thermal_dose: 240.0, // CEM43 for brain tissue
            min_skull_distance: 5.0, // mm
        }
    }
}

/// Transducer array specifications
#[derive(Debug, Clone)]
pub struct TransducerSpecification {
    pub num_elements: usize,
    pub frequency: f64,
    pub focal_distance: f64,
    pub radius: f64,
    pub sound_speed: f64,
}

impl Default for TransducerSpecification {
    fn default() -> Self {
        Self {
            num_elements: 1024,
            frequency: 650e3,      // 650 kHz for brain therapy
            focal_distance: 120.0, // mm
            radius: 80.0,          // mm
            sound_speed: 1500.0,   // m/s
        }
    }
}

/// Skull acoustic properties derived from CT
#[derive(Debug)]
pub struct SkullProperties {
    pub sound_speed: Array3<f64>,
    pub density: Array3<f64>,
    pub attenuation: Array3<f64>,
}
