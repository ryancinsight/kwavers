//! Time-domain delay reference policy and utilities.
//!
//! # ⚠️ DEPRECATED - This module has moved!
//!
//! **This module is deprecated and will be removed in v0.3.0.**
//!
//! ## New Location
//!
//! Use `crate::analysis::signal_processing::beamforming::time_domain::delay_reference` instead.
//!
//! This module has been moved as part of the architectural purification effort (ADR 003).
//! Signal processing algorithms belong in the **analysis layer**, not the **domain layer**.
//!
//! ## Migration Guide
//!
//! ### Old (Deprecated)
//! ```rust,ignore
//! use crate::domain::sensor::beamforming::time_domain::delay_reference::{
//!     DelayReference, relative_delays_s, alignment_shifts_s
//! };
//! ```
//!
//! ### New (Correct)
//! ```rust,ignore
//! use crate::analysis::signal_processing::beamforming::time_domain::{
//!     DelayReference, relative_delays_s, alignment_shifts_s
//! };
//! ```
//!
//! ## Why This Move?
//!
//! - **Layering**: Delay reference policy is an analysis decision, not a domain primitive
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
    note = "Moved to `crate::analysis::signal_processing::beamforming::time_domain::delay_reference`. \
            Update your imports to the new location."
)]
pub use crate::analysis::signal_processing::beamforming::time_domain::delay_reference::*;

// Keep original implementation as internal fallback (will be removed in v0.3.0)
#[allow(deprecated)]
use crate::core::error::{KwaversError, KwaversResult};

// All functionality has been moved to analysis::signal_processing::beamforming::time_domain::delay_reference
// Tests are now in the new location

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(deprecated)]
    fn deprecated_module_still_accessible() {
        // Verify deprecated re-exports still work for backward compatibility
        let delays = vec![0.010, 0.011, 0.009];
        let _tau_ref = DelayReference::SensorIndex(0)
            .resolve_reference_delay_s(&delays)
            .expect("ref");
        let _rel = relative_delays_s(&delays, DelayReference::SensorIndex(0)).expect("rel");
        let _shifts = alignment_shifts_s(&delays, DelayReference::SensorIndex(0)).expect("shifts");
    }
}
