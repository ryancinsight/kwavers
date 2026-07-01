//! Deterministic experiment artifact — immutable output bundle for one `run_experiment` call (Phase 5).
//!
//! [`ExperimentRecord`] is a pure-data struct: it carries the pre-step, the aggregated metrics,
//! and the beam `PhysicsReport` produced by a single `run_experiment` invocation. File emission
//! (`.kv`, `.npz`, BMP renders) is gated on the `io` Cargo feature and lives outside this struct;
//! `ExperimentRecord` itself has no I/O dependencies so it remains testable without file system
//! access.
//!
//! [`artifact_key`] produces a deterministic string key uniquely identifying the experiment
//! (frequency × lane-count × focal-depth) — suitable as a filename stem, a cache key, or a
//! log-line tag. It encodes the three axes that make two experiments physically distinct.

use crate::validate::{KwaversBeamStep, PhysicsReport};

use super::metrics::ExperimentMetrics;

/// Deterministic output bundle for one [`super::runner::run_experiment`] call.
///
/// Immutable after construction; `Clone` for test assertions and reporting pipelines.
#[derive(Debug, Clone)]
pub struct ExperimentRecord {
    /// The driver→transducer typed pre-step this experiment was run from.
    pub step: KwaversBeamStep,
    /// Aggregated acoustic + thermal metrics.
    pub metrics: ExperimentMetrics,
    /// 4-check kwavers-beam physics report (focal pressure / MI / grating / resistor margin).
    pub beam_report: PhysicsReport,
    /// True iff all 4 kwavers-beam checks passed (mirrors `beam_report.all_pass`).
    pub all_pass: bool,
}

impl ExperimentRecord {
    /// Assemble the record from its parts.
    #[must_use]
    pub fn new(
        step: KwaversBeamStep,
        metrics: ExperimentMetrics,
        beam_report: PhysicsReport,
    ) -> Self {
        let all_pass = beam_report.all_pass;
        Self {
            step,
            metrics,
            beam_report,
            all_pass,
        }
    }
}

/// Deterministic string key uniquely identifying an experiment from the three axes that make two
/// experiments physically distinct: drive frequency (MHz, 3 d.p.), lane count, and focal depth
/// (mm, 2 d.p.).
///
/// Example: `"500.000MHz_96ch_10.00mm"` for the article-class v2 stack.
#[must_use]
pub fn artifact_key(frequency_hz: f64, lanes: usize, focal_m: f64) -> String {
    let freq_mhz = frequency_hz * 1.0e-6;
    let focal_mm = focal_m * 1.0e3;
    format!("{freq_mhz:.3}MHz_{lanes}ch_{focal_mm:.2}mm")
}
