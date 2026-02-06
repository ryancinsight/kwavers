//! Conservation Violation Reports
//!
//! Generates detailed reports on conservation violations and conservation verification results.

use super::detectors::{ConservationViolation, ViolationStatistics};
use serde::{Deserialize, Serialize};
use std::fmt::Write as FmtWrite;

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

impl ConservationReport {
    /// Create a new conservation report
    pub fn new(
        _title: String,
        metadata: ReportMetadata,
        violations: &[ConservationViolation],
        statistics: &ViolationStatistics,
        timesteps: u64,
    ) -> Self {
        let status = Self::determine_status(statistics);
        let analysis =
            ConservationViolationAnalysis::from_violations(violations, statistics, timesteps);
        let recommendations = Self::generate_recommendations(&analysis);

        Self {
            metadata,
            status,
            analysis,
            recommendations,
        }
    }

    /// Determine overall status from statistics
    fn determine_status(stats: &ViolationStatistics) -> ConservationStatus {
        if stats.total_violations == 0 {
            ConservationStatus::Excellent
        } else if stats.critical_violations == 0 && stats.average_relative_error < 1e-4 {
            ConservationStatus::Good
        } else if stats.critical_violations < 5 && stats.max_relative_error < 1e-2 {
            ConservationStatus::Warning
        } else {
            ConservationStatus::Critical
        }
    }

    /// Generate recommendations based on analysis
    fn generate_recommendations(analysis: &ConservationViolationAnalysis) -> Vec<String> {
        let mut recommendations = Vec::new();

        if analysis.statistics.total_violations == 0 {
            recommendations.push("✓ Excellent conservation - no action needed".to_string());
            return recommendations;
        }

        if analysis.statistics.trend_worsening {
            recommendations.push("⚠ Violations are increasing over time - consider:".to_string());
            recommendations.push("  1. Reducing timestep size (CFL number)".to_string());
            recommendations.push("  2. Using higher-order discretization schemes".to_string());
            recommendations.push("  3. Checking for numerical instabilities".to_string());
        }

        if analysis.statistics.max_error > 1e-3 {
            recommendations.push("⚠ Large conservation errors detected - consider:".to_string());
            recommendations.push("  1. Refining spatial grid".to_string());
            recommendations.push(
                "  2. Using conservative discretization (mass-conservative interpolation)"
                    .to_string(),
            );
            recommendations.push("  3. Checking boundary conditions".to_string());
        }

        if analysis.statistics.violation_frequency > 0.1 {
            recommendations.push("⚠ Frequent violations - consider:".to_string());
            recommendations.push("  1. Validating physical parameters".to_string());
            recommendations.push("  2. Checking source/sink terms".to_string());
            recommendations
                .push("  3. Using implicit time integration for better stability".to_string());
        }

        recommendations
    }

    /// Format report as human-readable text
    pub fn to_text(&self) -> String {
        let mut output = String::new();

        writeln!(
            &mut output,
            "╔════════════════════════════════════════════════════════════════╗"
        )
        .ok();
        writeln!(
            &mut output,
            "║          CONSERVATION LAW VERIFICATION REPORT                  ║"
        )
        .ok();
        writeln!(
            &mut output,
            "╚════════════════════════════════════════════════════════════════╝"
        )
        .ok();
        writeln!(&mut output).ok();

        writeln!(&mut output, "Title: {}", self.metadata.title).ok();
        writeln!(&mut output, "Generated: {}", self.metadata.generated_at).ok();
        writeln!(&mut output).ok();

        writeln!(&mut output, "SIMULATION PARAMETERS").ok();
        writeln!(&mut output, "────────────────────").ok();
        writeln!(
            &mut output,
            "Time range: {:.3e} - {:.3e} s",
            self.metadata.time_start, self.metadata.time_end
        )
        .ok();
        writeln!(&mut output, "Timesteps: {}", self.metadata.timesteps).ok();
        writeln!(
            &mut output,
            "Grid: {} × {} × {}",
            self.metadata.grid_nx, self.metadata.grid_ny, self.metadata.grid_nz
        )
        .ok();
        writeln!(&mut output, "Tolerance: {:.3e}", self.metadata.tolerance).ok();
        writeln!(&mut output).ok();

        writeln!(&mut output, "OVERALL STATUS: {}", self.status).ok();
        writeln!(&mut output).ok();

        writeln!(&mut output, "CONSERVATION STATISTICS").ok();
        writeln!(&mut output, "──────────────────────").ok();
        writeln!(
            &mut output,
            "Total violations: {}",
            self.analysis.statistics.total_violations
        )
        .ok();
        writeln!(
            &mut output,
            "Critical violations: {}",
            self.analysis.statistics.critical_count
        )
        .ok();
        writeln!(
            &mut output,
            "Violation frequency: {:.2}%",
            self.analysis.statistics.violation_frequency * 100.0
        )
        .ok();
        writeln!(
            &mut output,
            "Max relative error: {:.3e}",
            self.analysis.statistics.max_error
        )
        .ok();
        writeln!(
            &mut output,
            "Avg error (violations only): {:.3e}",
            self.analysis.statistics.avg_violation_error
        )
        .ok();
        writeln!(
            &mut output,
            "Trend: {}",
            if self.analysis.statistics.trend_worsening {
                "WORSENING"
            } else {
                "STABLE"
            }
        )
        .ok();
        writeln!(&mut output).ok();

        if !self.analysis.worst_violations.is_empty() {
            writeln!(&mut output, "WORST VIOLATIONS").ok();
            writeln!(&mut output, "────────────────").ok();
            for (quantity, error) in self.analysis.worst_violations.iter().take(5) {
                writeln!(&mut output, "  {}: {:.3e}", quantity, error).ok();
            }
            writeln!(&mut output).ok();
        }

        writeln!(&mut output, "RECOMMENDATIONS").ok();
        writeln!(&mut output, "───────────────").ok();
        for rec in &self.recommendations {
            writeln!(&mut output, "{}", rec).ok();
        }

        output
    }
}

impl ConservationViolationAnalysis {
    /// Build analysis from violations and statistics
    fn from_violations(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conservation_status_display() {
        assert_eq!(ConservationStatus::Excellent.to_string(), "Excellent");
        assert_eq!(ConservationStatus::Good.to_string(), "Good");
        assert_eq!(ConservationStatus::Warning.to_string(), "Warning");
        assert_eq!(ConservationStatus::Critical.to_string(), "Critical");
    }

    #[test]
    fn test_report_generation() {
        let metadata = ReportMetadata {
            title: "Test Simulation".to_string(),
            time_start: 0.0,
            time_end: 1.0,
            timesteps: 100,
            grid_nx: 256,
            grid_ny: 256,
            grid_nz: 256,
            tolerance: 1e-6,
            generated_at: "2026-01-29T00:00:00Z".to_string(),
        };

        let violations = vec![];
        let statistics = ViolationStatistics {
            total_violations: 0,
            critical_violations: 0,
            max_relative_error: 0.0,
            average_relative_error: 0.0,
            violations_per_law: std::collections::HashMap::new(),
        };

        let report =
            ConservationReport::new("Test".to_string(), metadata, &violations, &statistics, 100);

        assert_eq!(report.status, ConservationStatus::Excellent);
        let text = report.to_text();
        assert!(text.contains("Excellent"));
    }
}
