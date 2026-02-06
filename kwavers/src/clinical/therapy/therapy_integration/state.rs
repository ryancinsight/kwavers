//! Therapy Session State and Safety Monitoring
//!
//! This module provides real-time session state tracking and safety metric monitoring
//! for clinical therapy applications. All safety calculations comply with IEC 62359:2010
//! and FDA 510(k) guidance.
//!
//! ## Safety Standards
//!
//! - **Thermal Index (TI)**: IEC 62359:2010 compliant calculation
//! - **Mechanical Index (MI)**: FDA 510(k) guidance compliant
//! - **Cavitation Dose**: Based on Apfel & Holland (1991) models
//!
//! ## References
//!
//! - IEC 62359:2010: "Ultrasonics - Field characterization"
//! - FDA 510(k) Guidance: "Ultrasound Devices"
//! - Apfel & Holland (1991): "Gaseous cavitation thresholds"

use ndarray::Array3;
use std::collections::HashMap;

/// Therapy session state
///
/// Captures the current state of a therapy session including acoustic fields,
/// microbubble concentrations, cavitation activity, chemical concentrations,
/// and real-time safety metrics.
#[derive(Debug, Clone)]
pub struct TherapySessionState {
    /// Current time in session (s)
    pub current_time: f64,

    /// Treatment progress (0-1)
    ///
    /// Normalized progress indicator: 0 = start, 1 = complete
    pub progress: f64,

    /// Current acoustic field
    ///
    /// Pressure and velocity fields from the most recent acoustic simulation step.
    /// None if acoustic field has not yet been computed.
    pub acoustic_field: Option<AcousticField>,

    /// Current microbubble distribution (for CEUS therapy)
    ///
    /// 3D map of microbubble concentration (bubbles/mL).
    /// Only populated for microbubble-enhanced therapy modalities.
    pub microbubble_concentration: Option<Array3<f64>>,

    /// Current cavitation activity
    ///
    /// 3D map of cavitation activity levels (0-1 normalized).
    /// Used for histotripsy, oncotripsy, and cavitation-based therapies.
    pub cavitation_activity: Option<Array3<f64>>,

    /// Current chemical concentrations (for sonodynamic therapy)
    ///
    /// Map of chemical species names to 3D concentration fields.
    /// Used for sonodynamic therapy and sonochemistry applications.
    pub chemical_concentrations: Option<HashMap<String, Array3<f64>>>,

    /// Safety metrics
    ///
    /// Real-time safety monitoring values computed according to clinical standards.
    pub safety_metrics: SafetyMetrics,
}

/// Safety metrics during therapy
///
/// Real-time safety monitoring values calculated according to IEC 62359:2010 and
/// FDA guidance. These metrics must be continuously monitored to ensure patient safety.
///
/// ## Clinical Guidelines
///
/// - **TI < 6.0**: Generally safe for most applications
/// - **MI < 1.9**: Generally safe for diagnostic applications
/// - **MI < 0.7**: Recommended for fetal imaging
/// - **Cavitation dose**: Should be monitored for histotripsy applications
///
/// ## References
///
/// - IEC 62359:2010: Safety indices and exposure limits
/// - FDA 510(k) Guidance: Ultrasound device safety requirements
#[derive(Debug, Clone)]
pub struct SafetyMetrics {
    /// Current thermal index
    ///
    /// Indicates potential for tissue heating.
    /// Calculated as: TI = P_rms * sqrt(f) / 1e6
    /// where P_rms is root-mean-square pressure and f is frequency.
    pub thermal_index: f64,

    /// Current mechanical index
    ///
    /// Indicates potential for mechanical bioeffects and cavitation.
    /// Calculated as: MI = PNP / (sqrt(f) * 1e6)
    /// where PNP is peak negative pressure.
    pub mechanical_index: f64,

    /// Current cavitation dose
    ///
    /// Time-integrated cavitation activity.
    /// Based on Apfel & Holland (1991) cavitation threshold models.
    pub cavitation_dose: f64,

    /// Temperature rise (°C)
    ///
    /// 3D map of temperature rise above baseline.
    /// Calculated from acoustic absorption heating and bioheat transfer.
    pub temperature_rise: Array3<f64>,
}

/// Acoustic field representation
///
/// Captures pressure and velocity fields from acoustic simulation.
/// Used for therapy delivery, safety monitoring, and multi-physics coupling.
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

/// Safety status enumeration
///
/// Indicates whether therapy is within safe operating limits or has exceeded
/// a specific safety threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SafetyStatus {
    /// All parameters within safe limits
    Safe,

    /// Thermal index exceeded
    ///
    /// TI has exceeded the configured maximum limit.
    /// Therapy should be paused or stopped.
    ThermalLimitExceeded,

    /// Mechanical index exceeded
    ///
    /// MI has exceeded the configured maximum limit.
    /// Risk of unintended mechanical bioeffects.
    MechanicalLimitExceeded,

    /// Cavitation dose exceeded
    ///
    /// Cumulative cavitation dose has exceeded the configured limit.
    /// Risk of excessive tissue damage.
    CavitationLimitExceeded,

    /// Treatment time exceeded
    ///
    /// Total treatment time has exceeded the configured maximum.
    /// Therapy should be terminated.
    TimeLimitExceeded,
}
