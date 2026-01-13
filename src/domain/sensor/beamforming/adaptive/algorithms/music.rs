//! MUSIC (Multiple Signal Classification) algorithm
//!
//! # ⚠️ DEPRECATED - This module has moved!
//!
//! **This module is deprecated and will be removed in v0.3.0.**
//!
//! ## New Location
//!
//! Use `crate::analysis::signal_processing::beamforming::adaptive::music` instead.
//!
//! This module has been moved as part of the architectural purification effort.
//! Signal processing algorithms belong in the **analysis layer**, not the **domain layer**.
//!
//! ## Migration Guide
//!
//! ### Old (Deprecated)
//! ```rust,ignore
//! use crate::domain::sensor::beamforming::adaptive::algorithms::MUSIC;
//! ```
//!
//! ### New (Correct)
//! ```rust,ignore
//! use crate::analysis::signal_processing::beamforming::adaptive::MUSIC;
//! ```
//!
//! ## Why This Move?
//!
//! - **Layering**: MUSIC is a signal processing algorithm, not a domain primitive
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
    note = "Moved to `crate::analysis::signal_processing::beamforming::adaptive::music`. \
            Update your imports to the new location."
)]
pub use crate::analysis::signal_processing::beamforming::adaptive::music::MUSIC;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(deprecated)]
    fn deprecated_module_still_accessible() {
        // Verify deprecated re-export still works for backward compatibility
        let _music = MUSIC::new(1);
    }
}
