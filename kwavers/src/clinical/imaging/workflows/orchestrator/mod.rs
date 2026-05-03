//! Real-time clinical workflow orchestration.
//!
//! Coordinates multi-modal data acquisition, real-time processing,
//! fusion, AI-enhanced analysis, and clinical report generation.

pub mod monitor;
pub mod workflow;

pub use monitor::PerformanceMonitor;
pub use workflow::ClinicalWorkflowOrchestrator;
