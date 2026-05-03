//! Conservation report data types and analysis builder.

use super::super::detectors::{ConservationViolation, ViolationStatistics};
use serde::{Deserialize, Serialize};

/// Detailed conservation verification report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationReport {
    /// Simulation metadata
    pub metadata: ReportMetadata,

    /// Overall conservation status
    pub status: ConservationStatus,

    /// Violation analysis
    pub analysis: ConservationViolationAnalysis,

    /// Recommendations for remediation
    pub recommendations: Vec<String>,
}

/// Report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    /// Simulation title
    pub title: String,

    /// Simulation time range
    pub time_start: f64,
    pub time_end: f64,

    /// Number of timesteps
    pub timesteps: u64,

    /// Grid dimensions
    pub grid_nx: usize,
    pub grid_ny: usize,
    pub grid_nz: usize,

    /// Conservation tolerance used
    pub tolerance: f64,

    /// Report generation time (RFC 3339 format)
    pub generated_at: String,
}

/// Overall conservation status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConservationStatus {
    /// All quantities conserved within tolerance
    Excellent,

    /// Minor violations detected but acceptable
    Good,

    /// Significant violations present
    Warning,

    /// Critical violations affecting reliability
    Critical,
}

impl std::fmt::Display for ConservationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Excellent => write!(f, "Excellent"),
            Self::Good => write!(f, "Good"),
            Self::Warning => write!(f, "Warning"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// Detailed analysis of conservation violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationViolationAnalysis {
    /// Statistics summary
    pub statistics: AnalysisStatistics,

    /// Violations by law type
    pub violations_by_law: std::collections::HashMap<String, LawAnalysis>,

    /// Timeline of violations
    pub violation_timeline: Vec<ViolationTimelineEntry>,

    /// Worst violations observed
    pub worst_violations: Vec<(String, f64)>,
}

/// Statistics from violation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisStatistics {
    /// Total violations detected
    pub total_violations: usize,

    /// Critical violations (severity > 0.8)
    pub critical_count: usize,

    /// Percentage of timesteps with violations
    pub violation_frequency: f64,

    /// Maximum relative error observed
    pub max_error: f64,

    /// Average relative error in violation timesteps
    pub avg_violation_error: f64,

    /// Trend (true = worsening, false = stable/improving)
    pub trend_worsening: bool,
}

/// Analysis for a specific conservation law
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LawAnalysis {
    /// Number of violations of this law
    pub violation_count: usize,

    /// Percentage of violations for this law
    pub percentage: f64,

    /// Maximum error for this law
    pub max_error: f64,

    /// Average error for this law
    pub avg_error: f64,

    /// Timesteps with violations
    pub violation_timesteps: Vec<u64>,
}

/// Single entry in violation timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationTimelineEntry {
    /// Timestep number
    pub timestep: u64,

    /// Simulation time
    pub time: f64,

    /// Quantity name
    pub quantity: String,

    /// Relative error
    pub error: f64,

    /// Severity (0.0-1.0)
    pub severity: f64,
}

impl ConservationViolationAnalysis {
    /// Build analysis from violations and statistics
    pub(super) fn from_violations(
        violations: &[ConservationViolation],
        stats: &ViolationStatistics,
        total_timesteps: u64,
    ) -> Self {
        let mut violations_by_law = std::collections::HashMap::new();
        let mut violation_timeline = Vec::new();
        let mut worst_violations = Vec::new();

        // Build per-law analysis
        for (law, count) in &stats.violations_per_law {
            let law_violations: Vec<_> = violations.iter().filter(|v| v.law == *law).collect();

            let max_error = law_violations
                .iter()
                .map(|v| v.relative_error)
                .fold(0.0, f64::max);

            let avg_error = if !law_violations.is_empty() {
                law_violations.iter().map(|v| v.relative_error).sum::<f64>()
                    / law_violations.len() as f64
            } else {
                0.0
            };

            let violation_timesteps = law_violations.iter().map(|v| v.timestep).collect();

            violations_by_law.insert(
                format!("{}", law),
                LawAnalysis {
                    violation_count: *count,
                    percentage: (*count as f64 / stats.total_violations as f64) * 100.0,
                    max_error,
                    avg_error,
                    violation_timesteps,
                },
            );
        }

        // Build timeline
        for v in violations {
            violation_timeline.push(ViolationTimelineEntry {
                timestep: v.timestep,
                time: v.time,
                quantity: v.quantity.clone(),
                error: v.relative_error,
                severity: v.severity,
            });
            worst_violations.push((v.quantity.clone(), v.relative_error));
        }

        // Sort worst violations
        worst_violations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Self {
            statistics: AnalysisStatistics {
                total_violations: stats.total_violations,
                critical_count: violations.iter().filter(|v| v.severity > 0.8).count(),
                violation_frequency: if total_timesteps > 0 {
                    (violations.len() as f64) / (total_timesteps as f64)
                } else {
                    0.0
                },
                max_error: stats.max_relative_error,
                avg_violation_error: stats.average_relative_error,
                trend_worsening: false, // Would need more history
            },
            violations_by_law,
            violation_timeline,
            worst_violations: worst_violations.into_iter().take(10).collect(),
        }
    }
}
