//! Transcranial Focused Ultrasound (tFUS) Implementation
//!
//! Complete implementation for non-invasive brain therapy and stimulation
//! including aberration correction, treatment planning, and safety monitoring.
//!
//! ## Overview
//!
//! Transcranial focused ultrasound enables precise, non-invasive brain interventions:
//! - **Therapeutic**: Essential tremor, Parkinson's, tumor ablation
//! - **Blood-Brain Barrier Opening**: Drug delivery enhancement
//! - **Neuromodulation**: Brain stimulation and network modulation
//!
//! ## Key Components
//!
//! 1. **CT-Based Treatment Planning**: Patient-specific skull modeling
//! 2. **Aberration Correction**: Phase conjugation for skull compensation
//! 3. **Safety Monitoring**: Thermal and mechanical index control
//! 4. **Real-time Targeting**: Motion compensation and beam steering
//!
//! ## References
//!
//! - Aubry, J. F., et al. (2003). "Experimental demonstration of noninvasive
//!   transskull adaptive focusing." *JASA*, 113(1), 84-93.
//! - Clement, G. T., & Hynynen, K. (2002). "A non-invasive method for focusing
//!   ultrasound through the skull." *PMB*, 47(8), 1219-1235.
//! - McDannold, N., et al. (2010). "MRI monitoring of heating produced by ultrasound
//!   absorption in the skull." *Urology*, 76(5), 1328-1331.

pub mod aberration_correction;
pub mod bbb_opening;
pub mod safety_monitoring;
pub mod treatment_planning;

pub use aberration_correction::{PhaseCorrection, TranscranialAberrationCorrection};
pub use bbb_opening::{BBBOpening, PermeabilityEnhancement};
pub use safety_monitoring::{MechanicalIndex, SafetyMonitor, ThermalDose};
pub use treatment_planning::{TargetVolume, TreatmentPlan, TreatmentPlanner};
