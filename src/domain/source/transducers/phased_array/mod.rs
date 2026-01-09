//! Phased Array Transducer Implementation
//!
//! Electronic beam steering and focusing capabilities with realistic element modeling.
//!
//! # Architecture
//! - Modular design following SOLID principles
//! - Clear separation between configuration, elements, beamforming, and physics
//!
//! # References
//! - Szabo, T. L. (2014). "Diagnostic Ultrasound Imaging: Inside Out"
//! - Jensen, J. A. (1996). "Field: A program for simulating ultrasound systems"

pub mod beamforming;
pub mod config;
pub mod crosstalk;
pub mod element;
pub mod transducer;

pub use beamforming::BeamformingMode;
pub use config::PhasedArrayConfig;
pub use element::{ElementSensitivity, TransducerElement};
pub use transducer::PhasedArrayTransducer;
