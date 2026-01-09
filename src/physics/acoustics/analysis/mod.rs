//! Field Analysis and Beam Pattern Calculation Module
//!
//! This module provides comprehensive field analysis tools for beam pattern
//! calculation and field analysis functions.
//!
//! # Design Principles
//! - **SOLID**: Single responsibility for field analysis functions
//! - **DRY**: Reusable field analysis implementations
//! - **Zero-Copy**: Uses `ArrayView` and efficient iterators
//! - **KISS**: Clear, well-documented field analysis interfaces
//!
//! # Literature References
//! - O'Neil (1949): "Theory of focusing radiators"
//! - Cobbold (2007): "Foundations of Biomedical Ultrasound"
//! - Treeby & Cox (2010): "MATLAB toolbox"
//! - Szabo (2014): "Diagnostic Ultrasound Imaging"

pub mod beam_pattern;
pub mod beam_patterns;
pub mod focus;
pub mod metrics;
pub mod pressure;

pub use beam_pattern::{
    calculate_beam_pattern, calculate_directivity, BeamPatternConfig, FarFieldMethod,
};
pub use beam_patterns::BeamPatterns;
pub use focus::{calculate_beam_width, find_focal_plane, find_focus};
pub use metrics::{calculate_field_metrics, find_peak_pressure, FieldMetrics};
pub use pressure::{calculate_intensity, calculate_mechanical_index, calculate_thermal_index};
