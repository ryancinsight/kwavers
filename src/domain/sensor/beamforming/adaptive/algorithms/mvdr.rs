//! Minimum Variance Distortionless Response (MVDR/Capon) beamformer
//!
//! # ⚠️ DEPRECATED - This module has moved!
//!
//! **This module is deprecated and will be removed in v0.3.0.**
//!
//! ## New Location
//!
//! Use `crate::analysis::signal_processing::beamforming::adaptive::mvdr` instead.
//!
//! This module has been moved as part of the architectural purification effort.
//! Signal processing algorithms belong in the **analysis layer**, not the **domain layer**.
//!
//! ## Migration Guide
//!
//! ### Old (Deprecated)
//! ```rust,ignore
//! use crate::domain::sensor::beamforming::adaptive::algorithms::MinimumVariance;
//! ```
//!
//! ### New (Correct)
//! ```rust,ignore
//! use crate::analysis::signal_processing::beamforming::adaptive::MinimumVariance;
//! ```
//!
//! ## Why This Move?
//!
//! - **Layering**: MVDR is a signal processing algorithm, not a domain primitive
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
    note = "Moved to `crate::analysis::signal_processing::beamforming::adaptive::mvdr`. \
            Update your imports to the new location."
)]
pub use crate::analysis::signal_processing::beamforming::adaptive::mvdr::MinimumVariance;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(deprecated)]
    fn deprecated_module_still_accessible() {
        // Verify deprecated re-export still works for backward compatibility
        let _mvdr = MinimumVariance::default();
    }
}
