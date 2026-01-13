// Simple config structs for clinical workflows
#[derive(Debug, Clone)]
pub struct PhotoacousticConfig {
    pub _wavelength: f64,
    pub _optical_energy: f64,
    pub _absorption_coefficient: f64,
    pub _speed_of_sound: f64,
    pub _sampling_frequency: f64,
    pub _num_detectors: usize,
    pub _detector_radius: f64,
    pub _center_frequency: f64,
}

#[derive(Debug, Clone)]
pub struct ElastographyConfig {
    pub _excitation_frequency: f64,
    pub _push_duration: f64,
    pub _track_duration: f64,
    pub _push_focal_depth: f64,
    pub _track_focal_depth: f64,
    pub _frame_rate: f64,
    pub _num_tracking_beams: usize,
}

/// Clinical workflow configuration
#[derive(Debug, Clone)]
pub struct ClinicalWorkflowConfig {
    /// Target application (oncology, cardiology, etc.)
    pub application: ClinicalApplication,
    /// Priority level for resource allocation
    pub priority: WorkflowPriority,
    /// Quality vs speed trade-off
    pub quality_preference: QualityPreference,
    /// Enable real-time processing
    pub real_time_enabled: bool,
    /// Maximum acceptable latency (ms)
    pub max_latency_ms: u64,
    /// Enable AI decision support
    pub ai_decision_support: bool,
    /// Clinical protocol to follow
    pub protocol: ClinicalProtocol,
}

impl Default for ClinicalWorkflowConfig {
    fn default() -> Self {
        Self {
            application: ClinicalApplication::General,
            priority: WorkflowPriority::Standard,
            quality_preference: QualityPreference::Balanced,
            real_time_enabled: true,
            max_latency_ms: 500, // 500ms max latency
            ai_decision_support: true,
            protocol: ClinicalProtocol::Standard,
        }
    }
}

/// Clinical applications
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClinicalApplication {
    /// General diagnostic imaging
    General,
    /// Oncology imaging and biopsy guidance
    Oncology,
    /// Cardiac imaging and assessment
    Cardiology,
    /// Neurological imaging
    Neurology,
    /// Musculoskeletal imaging
    Musculoskeletal,
    /// Vascular imaging
    Vascular,
}

/// Workflow priority levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkflowPriority {
    /// Emergency/critical care
    Critical,
    /// High priority examination
    High,
    /// Standard clinical workflow
    Standard,
    /// Low priority screening
    Low,
}

/// Quality vs speed preference
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QualityPreference {
    /// Maximum image quality (slower processing)
    Quality,
    /// Balanced quality and speed
    Balanced,
    /// Maximum speed (reduced quality)
    Speed,
}

/// Clinical protocols
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClinicalProtocol {
    /// Standard diagnostic protocol
    Standard,
    /// Research protocol with extended capabilities
    Research,
    /// Screening protocol optimized for speed
    Screening,
    /// Interventional protocol for procedures
    Interventional,
}
