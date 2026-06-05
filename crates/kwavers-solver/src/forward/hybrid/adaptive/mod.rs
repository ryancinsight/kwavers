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
pub mod statistics;

pub use criteria::HybridAdaptiveSelectionCriteria;
pub use metrics::{DetailedMetrics, HybridAdaptiveQualityMetrics};
pub use crate::hybrid::adaptive_selection::AdaptiveSelector;
pub use statistics::{FrequencySpectrum, StatisticalMetrics};
