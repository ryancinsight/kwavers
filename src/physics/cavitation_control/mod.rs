//! Cavitation Control Module
//!
//! Implements amplitude/power modulation techniques for cavitation control
//! using negative feedback loops.
//!
//! Literature references:
//! - Coussios & Roy (2008): "Applications of acoustics and cavitation to noninvasive therapy and drug delivery"
//! - Gyöngy & Coussios (2010): "Passive spatial mapping of inertial cavitation"
//! - Hockham et al. (2010): "A real-time controller for sustaining thermally relevant acoustic cavitation"
//! - Arvanitis et al. (2012): "Controlled ultrasound-induced blood-brain barrier disruption"

pub mod cavitation_detector;
pub mod detection; // Modular detection components
pub mod feedback_controller;
pub mod pid_controller;
pub mod power_modulation;

pub use feedback_controller::{
    CavitationMetrics, ControlOutput, ControlStrategy, FeedbackConfig, FeedbackController,
};

pub use power_modulation::{
    AmplitudeController, DutyCycleController, ModulationScheme, PowerControl, PowerModulator,
};

pub use cavitation_detector::{
    BroadbandDetector, CavitationDetector, CavitationState, DetectionMethod, SpectralDetector,
    SubharmonicDetector,
};

pub use pid_controller::{ControllerOutput, ErrorIntegral, PIDConfig, PIDController, PIDGains};
