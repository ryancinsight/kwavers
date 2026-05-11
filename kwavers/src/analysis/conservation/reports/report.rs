//! ConservationReport construction and formatting.

use super::super::detectors::{ConservationViolation, ViolationStatistics};
use super::types::{
    ConservationReport, ConservationStatus, ConservationViolationAnalysis, ReportMetadata,
};
use std::fmt::Write as FmtWrite;

impl ConservationReport {
    /// Create a new conservation report
    #[must_use] 
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
            recommendations.push("✓ Excellent conservation - no action needed".to_owned());
            return recommendations;
        }

        if analysis.statistics.trend_worsening {
            recommendations.push("⚠ Violations are increasing over time - consider:".to_owned());
            recommendations.push("  1. Reducing timestep size (CFL number)".to_owned());
            recommendations.push("  2. Using higher-order discretization schemes".to_owned());
            recommendations.push("  3. Checking for numerical instabilities".to_owned());
        }

        if analysis.statistics.max_error > 1e-3 {
            recommendations.push("⚠ Large conservation errors detected - consider:".to_owned());
            recommendations.push("  1. Refining spatial grid".to_owned());
            recommendations.push(
                "  2. Using conservative discretization (mass-conservative interpolation)".to_owned(),
            );
            recommendations.push("  3. Checking boundary conditions".to_owned());
        }

        if analysis.statistics.violation_frequency > 0.1 {
            recommendations.push("⚠ Frequent violations - consider:".to_owned());
            recommendations.push("  1. Validating physical parameters".to_owned());
            recommendations.push("  2. Checking source/sink terms".to_owned());
            recommendations
                .push("  3. Using implicit time integration for better stability".to_owned());
        }

        recommendations
    }

    /// Format report as human-readable text
    #[must_use] 
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
