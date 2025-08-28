// solver/hybrid/adaptive_selection/mod.rs - Modular adaptive selection

pub mod analyzer;
pub mod criteria;
pub mod metrics;
pub mod selector;
pub mod statistics;

// Re-export main types
pub use analyzer::FieldAnalyzer;
pub use criteria::SelectionCriteria;
pub use metrics::{ComputationalMetrics, MaterialMetrics, SpectralMetrics};
pub use selector::{AdaptiveMethodSelector, AdaptiveSelector};
pub use statistics::SelectionStatistics;
