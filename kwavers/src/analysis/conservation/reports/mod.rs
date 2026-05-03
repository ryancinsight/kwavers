//! Conservation Violation Reports
//!
//! Generates detailed reports on conservation violations and conservation verification results.

mod report;
#[cfg(test)]
mod tests;
pub mod types;

pub use types::{
    AnalysisStatistics, ConservationReport, ConservationStatus, ConservationViolationAnalysis,
    LawAnalysis, ReportMetadata, ViolationTimelineEntry,
};
