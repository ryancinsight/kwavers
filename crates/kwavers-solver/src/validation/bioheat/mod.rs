//! Bioheat Validation Module
//!
//! Provides validation of bioheat transfer calculations and CEM43 thermal dose
//! metrics against published literature references.
//!
//! # Module Structure
//!
//! - `cem43_reference`: Analytical reference solutions for CEM43 calculations
//! - Validation test suites for literature comparison

pub mod cem43_reference;

pub use cem43_reference::{
    analytical_cem43_constant, analytical_cem43_ramp, literature_cases, r_factor_at_temperature,
    time_for_target_cem43, R_FACTOR_SUBTHRESHOLD, R_FACTOR_SUPRATHRESHOLD,
    STANDARD_DAMAGE_THRESHOLD, THRESHOLD_TEMP_C,
};
