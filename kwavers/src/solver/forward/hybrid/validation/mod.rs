//! Hybrid solver validation framework
//!
//! Provides comprehensive validation and testing for hybrid PSTD/FDTD solvers

pub mod config;
pub mod metrics;
pub mod report;
pub mod suite;
pub mod test_cases;

// Re-export main types
pub use config::HybridValidationSuiteConfig;
pub use metrics::{ErrorBounds, ErrorMetrics, HybridSolverMetrics};
pub use report::{HybridValidationReport, ValidationSummary};
pub use suite::HybridValidationSuite;
pub use test_cases::{TestResult, ValidationTestCase};
