//! Passive Acoustic Mapping (PAM) for cavitation detection and sonoluminescence
//!
//! This module implements passive acoustic mapping techniques for detecting and
//! mapping cavitation fields and sonoluminescence events using arbitrary sensor
//! array geometries.
//!
//! ## Architectural Note (SSOT Enforcement)
//!
//! PAM provides **no** independent beamforming algorithm implementations.
//! Beamforming algorithms and numerical primitives are owned by
//! `crate::sensor::beamforming` (single source of truth). PAM owns:
//! - beamforming *policy* (method selection, apodization, focal point, bands)
//! - map construction and post-processing (TEA, band power integration, etc.)
//!
//! This preserves a deep vertical separation of concern.
//!
//! ## Literature References
//!
//! 1. **Gyöngy & Coussios (2010)**: "Passive spatial mapping of inertial cavitation
//!    during HIFU exposure", IEEE Trans. Biomed. Eng.
//! 2. **Haworth et al. (2012)**: "Passive imaging with pulsed ultrasound insonations",
//!    J. Acoust. Soc. Am.
//! 3. **Coviello et al. (2015)**: "Passive acoustic mapping utilizing optimal beamforming
//!    in ultrasound therapy monitoring", J. Acoust. Soc. Am.

pub mod geometry;
pub use geometry::{ArrayElement, ArrayGeometry, DirectivityPattern};
