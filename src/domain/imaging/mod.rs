//! Imaging Domain Module
//!
//! This module contains imaging-related domain types. Some types have been moved to the
//! clinical layer as part of Sprint 188 Phase 3 (Domain Layer Cleanup).
//!
//! ## Migration Notice
//!
//! **⚠️ IMPORTANT**: Photoacoustic imaging types have been relocated to enforce proper architectural layering.
//!
//! ### Photoacoustic Imaging
//!
//! **Old Location** (No Longer Valid):
//! ```rust,ignore
//! use crate::domain::imaging::photoacoustic::{PhotoacousticParameters, PhotoacousticResult};
//! ```
//!
//! **New Location** (Use This):
//! ```rust,ignore
//! use crate::clinical::imaging::photoacoustic::{PhotoacousticParameters, PhotoacousticResult};
//! ```
//!
//! ## Current Contents
//!
//! - `ultrasound` - Ultrasound imaging domain types (modes, configurations, modalities)
//!
//! ## See Also
//!
//! - `clinical::imaging` - Application-level imaging workflows and types
//! - `physics::foundations` - Physics specifications for wave equations
//! - `domain::sensor` - Sensor primitives for signal detection
//! - `domain::source` - Source primitives for wave generation

pub mod ultrasound;
