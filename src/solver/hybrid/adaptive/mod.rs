//! Adaptive Selection for Hybrid Numerical Methods
//!
//! Intelligent selection of optimal numerical methods based on:
//! - Field smoothness and spectral content
//! - Material properties and interfaces
//! - Computational efficiency metrics
//!
//! # Architecture
//! - Modular analysis components
//! - Composable selection criteria
//! - Statistical and frequency analysis

pub mod criteria;
pub mod metrics;
pub mod selector;
pub mod statistics;

pub use criteria::SelectionCriteria;
pub use metrics::{DetailedMetrics, QualityMetrics};
// Note: AdaptiveSelector moved to adaptive_selection module
pub use crate::solver::hybrid::adaptive_selection::AdaptiveSelector;
pub use statistics::{FrequencySpectrum, StatisticalMetrics};