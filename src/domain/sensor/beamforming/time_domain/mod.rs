#![deny(missing_docs)]
//! Time-domain beamforming (broadband / transient processing).
//!
//! # ⚠️ DEPRECATED - This module has moved!
//!
//! **This module is deprecated and will be removed in v0.3.0.**
//!
//! ## New Location
//!
//! Use `crate::analysis::signal_processing::beamforming::time_domain` instead.
//!
//! This module has been moved as part of the architectural purification effort (ADR 003).
//! Signal processing algorithms belong in the **analysis layer**, not the **domain layer**.
//!
//! ## Migration Guide
//!
//! ### Old (Deprecated)
//! ```rust,ignore
//! use crate::domain::sensor::beamforming::time_domain::{
//!     das::delay_and_sum_time_domain_with_reference,
//!     delay_reference::{DelayReference, relative_delays_s},
//!     DEFAULT_DELAY_REFERENCE,
//! };
//! ```
//!
//! ### New (Correct)
//! ```rust,ignore
//! use crate::analysis::signal_processing::beamforming::time_domain::{
//!     delay_and_sum,  // Note: renamed from delay_and_sum_time_domain_with_reference
//!     DelayReference,
//!     relative_delays_s,
//!     DEFAULT_DELAY_REFERENCE,
//! };
//! ```
//!
//! ## Why This Move?
//!
//! - **Layering**: Time-domain beamforming is a signal processing algorithm, not a domain primitive
//! - **Dependencies**: Analysis layer can depend on domain, but not vice versa
//! - **Reusability**: Beamforming should work on data from simulations, sensors, and clinical workflows
//! - **Literature Alignment**: Standard references treat beamforming as signal processing
//!
//! ## Backward Compatibility
//!
//! This module currently re-exports from the new location for backward compatibility.
//! Please update your code to use the new location before v0.3.0.
//!
//! ## Original Documentation (for reference)
//!
//! ### Field jargon
//! - **Time-domain DAS** is also called **conventional beamforming** or **shift-and-sum**.
//! - For transient localization the dominant pattern is **SRP-DAS** (Steered Response Power):
//!   evaluate candidate points by steering (TOF alignment) and scoring energy,
//!   e.g. `SRP(p) = ∑_t |y_p(t)|²` where `y_p(t)` is the steered DAS output.
//!
//! ### Module layout
//! - `das`: delay-and-sum (shift-and-sum) building blocks (DEPRECATED, moved to analysis layer)
//! - `delay_reference`: delay datum policies and TOF→relative-delay utilities (DEPRECATED, moved to analysis layer)

pub mod das;
pub mod delay_reference;

// Re-exports: keep domain terms discoverable at the `time_domain` level (DEPRECATED).
#[deprecated(
    since = "0.2.0",
    note = "Moved to `crate::analysis::signal_processing::beamforming::time_domain`. Update your imports."
)]
pub use delay_reference::{alignment_shifts_s, relative_delays_s, DelayReference};

#[deprecated(
    since = "0.2.0",
    note = "Moved to `crate::analysis::signal_processing::beamforming::time_domain::DEFAULT_DELAY_REFERENCE`"
)]
pub use das::DEFAULT_DELAY_REFERENCE;
