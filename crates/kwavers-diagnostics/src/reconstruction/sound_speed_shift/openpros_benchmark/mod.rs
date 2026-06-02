//! OpenPros-style limited-view speed-shift benchmark fixture.
//!
//! The benchmark owns only the deterministic phantom, top/bottom probe
//! acquisition geometry, waveform metadata, and comparison metrics. Dense and
//! sparse reconstructions still use `SoundSpeedShiftPlan`.

mod acquisition;
mod metrics;
mod phantom;
mod pipeline;
#[cfg(test)]
mod tests;
mod types;

pub use pipeline::{openpros_shift_benchmark_case, run_openpros_shift_benchmark};
pub use types::{
    OpenProsShiftBenchmarkCase, OpenProsShiftBenchmarkConfig, OpenProsShiftBenchmarkResult,
    OpenProsShiftReconstructionMetrics, OpenProsWaveformExpectation, OPENPROS_PAPER_ID,
};
