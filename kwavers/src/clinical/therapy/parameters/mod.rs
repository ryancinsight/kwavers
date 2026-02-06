//! Therapy Parameters Module
//!
//! This module provides configuration parameters for therapeutic ultrasound treatments.
//! It contains treatment settings, safety limits, and validation logic.
//!
//! ## Architecture
//!
//! This module resides in the **clinical/therapy** layer because therapy parameters
//! are application-level configuration, not domain primitives. They combine physics
//! constraints with clinical protocols and safety regulations.
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use kwavers::clinical::therapy::parameters::TherapyParameters;
//!
//! // Configure HIFU treatment
//! let params = TherapyParameters::hifu();
//! println!("Mechanical Index: {:.2}", params.mechanical_index);
//!
//! // Custom parameters
//! let mut custom = TherapyParameters::new(1.5e6, 2.0e6, 10.0);
//! if !custom.validate_safety() {
//!     eprintln!("Warning: Parameters exceed safety limits");
//! }
//! ```
//!
//! ## Migration Notice
//!
//! **⚠️ IMPORTANT**: This module was moved from `domain::therapy::parameters` to
//! `clinical::therapy::parameters` in Sprint 188 Phase 3 (Domain Layer Cleanup).
//!
//! ### Old Import (No Longer Valid)
//! ```rust,ignore
//! use crate::domain::therapy::parameters::TherapyParameters;
//! ```
//!
//! ### New Import (Correct Location)
//! ```rust,ignore
//! use crate::clinical::therapy::parameters::TherapyParameters;
//! ```

pub mod types;

pub use types::TherapyParameters;
