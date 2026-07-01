//! # Experiment framework (Phase 6).
//!
//! Orchestrates end-to-end driver→transducer simulation:
//! stim profile → dispatch → acoustic simulation → thermal propagation →
//! metrics aggregation → deterministic artifact record.
//!
//! Each sub-module owns a single trait or function-set; dependency injection is via
//! trait bounds so the orchestrator depends on no concrete simulator.
//!
//! # Slice layout
//!
//! ```text
//! src/experiment/
//! ├── mod.rs        ← this file (re-exports + public surface)
//! ├── stimulus.rs   ← Stimulus trait + DefaultStimulus (DIP seam, manifest-agnostic)
//! ├── acoustic.rs   ← AcousticSimulator trait + PressureMap + InCrateAcousticSim +
//! │                   (feature-gated) KwaversSim (kwavers-transducer propagation backend)
//! ├── thermal.rs    ← ThermalState + propagate_thermal
//! ├── dispatch.rs   ← LaneBinding + TileDispatch (lane→tile equal-partition)
//! ├── metrics.rs    ← ExperimentMetrics + build_beam_report
//! ├── recorder.rs   ← ExperimentRecord + artifact_key
//! ├── runner.rs     ← run_experiment + ExperimentReport
//! └── tests.rs      ← end-to-end + per-module value-semantic tests
//! ```
//!
//! # SSOT for the slice
//!
//! * [`stimulus::DefaultStimulus`] — borrows `manifest.tile_profiles` verbatim.
//! * [`acoustic::InCrateAcousticSim`] — the default; uses [`crate::physics::acoustic`] functions.
//! * `acoustic::KwaversSim` (feature `kwavers`) — calls `kwavers-transducer` to synthesize the
//!   realized channel geometry and propagate the focused pressure envelope.
//! * [`runner::run_experiment`] — the public entry point.

pub mod acoustic;
pub mod dispatch;
pub mod metrics;
pub mod recorder;
pub mod runner;
pub mod stimulus;
pub mod thermal;

#[cfg(test)]
mod tests;

pub use acoustic::{AcousticSimulator, InCrateAcousticSim, PressureMap};
pub use dispatch::{LaneBinding, TileDispatch};
pub use metrics::{build_beam_report, ExperimentMetrics};
pub use recorder::{artifact_key, ExperimentRecord};
pub use runner::{run_experiment, ExperimentReport};
pub use stimulus::{DefaultStimulus, Stimulus};
pub use thermal::{propagate_thermal, ThermalState};

#[cfg(feature = "kwavers")]
pub use acoustic::KwaversSim;
