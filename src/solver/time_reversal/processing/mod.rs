//! Signal Processing Module for Time-Reversal
//!
//! Provides filtering, amplitude correction, and windowing functions.

pub mod amplitude;
pub mod filters;
pub mod windowing;

pub use amplitude::AmplitudeCorrector;
pub use filters::FrequencyFilter;
pub use windowing::{apply_spatial_window, tukey_window};
