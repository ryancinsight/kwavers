#![deny(missing_docs)]
//! Time-domain beamforming (broadband / transient processing).
//!
//! # Field jargon
//! - **Time-domain DAS** is also called **conventional beamforming** or **shift-and-sum**.
//! - For transient localization the dominant pattern is **SRP-DAS** (Steered Response Power):
//!   evaluate candidate points by steering (TOF alignment) and scoring energy,
//!   e.g. `SRP(p) = ∑_t |y_p(t)|²` where `y_p(t)` is the steered DAS output.
//!
//! # Architectural intent (SSOT / deep vertical tree)
//! - Algorithmic primitives live in `crate::sensor::beamforming`.
//! - This subtree isolates time-domain (broadband) operations from narrowband/adaptive methods.
//!
//! # Next steps (recommended defaults)
//! The default delay reference for localization should be a **fixed reference sensor**,
//! commonly element 0 (`SensorIndex(0)`). This makes the delay datum explicit and keeps
//! SRP-DAS scoring point-dependent for transient data models.
//!
//! ## Module layout
//! - `das`: delay-and-sum (shift-and-sum) building blocks.
//! - `delay_reference`: delay datum policies and TOF→relative-delay utilities.

pub mod das;
pub mod delay_reference;

// Re-exports: keep domain terms discoverable at the `time_domain` level.
pub use delay_reference::{alignment_shifts_s, relative_delays_s, DelayReference};
