//! Therapy Metrics Module
//!
//! This module provides metrics for tracking therapeutic ultrasound treatment progress,
//! outcomes, and safety indicators. It includes thermal dose calculations (CEM43),
//! cavitation monitoring, and treatment efficiency metrics.
//!
//! ## Architecture
//!
//! This module resides in the **clinical/therapy** layer because treatment metrics
//! are application-level concerns that combine physics outputs with clinical protocols.
//! They are not domain primitives but rather derived measurements used for treatment
//! monitoring and safety assessment.
//!
//! ## Thermal Dose (CEM43)
//!
//! The Cumulative Equivalent Minutes at 43°C (CEM43) is the standard metric for
//! thermal ablation:
//!
//! ```text
//! CEM43 = ∫ R^(T(t) - 43) dt
//! ```
//!
//! where:
//! - R = 0.5 (2^(T-43)) for T ≥ 43°C (protein denaturation regime)
//! - R = 0.25 (4^(T-43)) for 37°C < T < 43°C (sub-lethal heating)
//! - T(t) = temperature time course (°C)
//!
//! **Clinical Thresholds**:
//! - CEM43 ≥ 240 min: Complete necrosis (ablation)
//! - CEM43 = 30-240 min: Partial necrosis
//! - CEM43 < 30 min: Reversible effects
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use kwavers::clinical::therapy::metrics::TreatmentMetrics;
//! use ndarray::Array3;
//!
//! let mut metrics = TreatmentMetrics::default();
//!
//! // Simulate treatment over time
//! for t in 0..100 {
//!     let temperature = Array3::<f64>::zeros((50, 50, 50)); // Replace with actual
//!     let cavitation = Array3::<f64>::zeros((50, 50, 50));
//!     let dt = 0.1; // 100 ms time step
//!
//!     // Accumulate doses
//!     let thermal_dose = TreatmentMetrics::calculate_thermal_dose(&temperature, dt);
//!     metrics.thermal_dose += thermal_dose;
//!
//!     let cav_dose = TreatmentMetrics::calculate_cavitation_dose(&cavitation, dt);
//!     metrics.cavitation_dose += cav_dose;
//!
//!     // Update peak temperature
//!     metrics.update_peak_temperature(&temperature);
//! }
//!
//! // Check if treatment was successful
//! let target_dose = 240.0; // CEM43 minutes
//! if metrics.is_successful(target_dose, 0.9) {
//!     println!("Treatment successful: {}", metrics.summary());
//! }
//! ```
//!
//! ## Migration Notice
//!
//! **⚠️ IMPORTANT**: This module was moved from `domain::therapy::metrics` to
//! `clinical::therapy::metrics` in Sprint 188 Phase 3 (Domain Layer Cleanup).
//!
//! ### Old Import (No Longer Valid)
//! ```rust,ignore
//! use crate::domain::therapy::metrics::TreatmentMetrics;
//! ```
//!
//! ### New Import (Correct Location)
//! ```rust,ignore
//! use crate::clinical::therapy::metrics::TreatmentMetrics;
//! ```
//!
//! ## Safety Considerations
//!
//! ### Thermal Safety
//! - Peak temperature < 100°C (avoid boiling)
//! - CEM43 < 240 min outside target zone
//! - Monitor near-field heating (transducer face)
//!
//! ### Cavitation Safety
//! - Mechanical Index (MI) < 1.9 for diagnostic
//! - Higher MI allowed for therapy with informed consent
//! - Monitor for unpredictable cavitation (hot spots)
//!
//! ## References
//!
//! - Sapareto, S. A., & Dewey, W. C. (1984). "Thermal dose determination in cancer therapy."
//!   *International Journal of Radiation Oncology Biology Physics*, 10(6), 787-800.
//! - Yarmolenko, P. S., et al. (2011). "Thresholds for thermal damage to normal tissues."
//!   *International Journal of Hyperthermia*, 27(4), 320-343.
//! - IEC 62359:2017 "Ultrasonics - Field characterization - Test methods for thermal index"

pub mod types;

pub use types::TreatmentMetrics;
