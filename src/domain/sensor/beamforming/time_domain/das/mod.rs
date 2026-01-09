//! Time-domain Delay-and-Sum (DAS) beamforming with explicit delay reference policy.
//!
//! # ⚠️ DEPRECATED - This module has moved!
//!
//! **This module is deprecated and will be removed in v0.3.0.**
//!
//! ## New Location
//!
//! Use `crate::analysis::signal_processing::beamforming::time_domain::das` instead.
//!
//! This module has been moved as part of the architectural purification effort (ADR 003).
//! Signal processing algorithms belong in the **analysis layer**, not the **domain layer**.
//!
//! ## Migration Guide
//!
//! ### Old (Deprecated)
//! ```rust,ignore
//! use crate::domain::sensor::beamforming::time_domain::das::{
//!     delay_and_sum_time_domain_with_reference, DEFAULT_DELAY_REFERENCE
//! };
//! ```
//!
//! ### New (Correct)
//! ```rust,ignore
//! use crate::analysis::signal_processing::beamforming::time_domain::{
//!     delay_and_sum, DEFAULT_DELAY_REFERENCE
//! };
//! ```
//!
//! **Note:** The function has been renamed from `delay_and_sum_time_domain_with_reference`
//! to simply `delay_and_sum` in the new location.
//!
//! ## Why This Move?
//!
//! - **Layering**: DAS is a signal processing algorithm, not a domain primitive
//! - **Dependencies**: Analysis layer can depend on domain, but not vice versa
//! - **Reusability**: Beamforming should work on data from simulations, sensors, and clinical workflows
//!
//! ## Backward Compatibility
//!
//! This module currently re-exports from the new location for backward compatibility.
//! Please update your code to use the new location before v0.3.0.

// Re-export from new location for backward compatibility
#[deprecated(
    since = "0.2.0",
    note = "Moved to `crate::analysis::signal_processing::beamforming::time_domain`. \
            Use `delay_and_sum` instead of `delay_and_sum_time_domain_with_reference`. \
            Update your imports to the new location."
)]
pub use crate::analysis::signal_processing::beamforming::time_domain::delay_and_sum as delay_and_sum_time_domain_with_reference;

#[deprecated(
    since = "0.2.0",
    note = "Moved to `crate::analysis::signal_processing::beamforming::time_domain::DEFAULT_DELAY_REFERENCE`"
)]
pub use crate::analysis::signal_processing::beamforming::time_domain::DEFAULT_DELAY_REFERENCE;

// All functionality has been moved to analysis::signal_processing::beamforming::time_domain::das
// Tests are now in the new location

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    #[allow(deprecated)]
    fn deprecated_module_still_accessible() {
        // Verify deprecated re-exports still work for backward compatibility
        let fs = 10.0;
        let n = 8usize;

        let delays = vec![1.0, 1.2];
        let weights = vec![1.0, 1.0];

        let mut x = Array3::<f64>::zeros((2, 1, n));
        x[[0, 0, 3]] = 1.0;
        x[[1, 0, 5]] = 1.0;

        let _y = delay_and_sum_time_domain_with_reference(
            &x,
            fs,
            &delays,
            &weights,
            crate::analysis::signal_processing::beamforming::time_domain::DelayReference::SensorIndex(0),
        )
        .expect("deprecated function should still work");
    }
}
