//! Transcranial full-wave inversion (FWI) example.
//!
//! The entry point is intentionally a manifest. The workflow, acquisition,
//! phantom construction, metrics, and fixed-grid configuration live in
//! single-concern modules under `transcranial_fwi/`.
//!
//! The example supports an explicit synthetic skull phantom or a real CT
//! input selected with `KWAVERS_SEISMIC_INPUT_MODE=ct:<path>`. It demonstrates
//! the acoustic L2 objective and adjoint-state gradient described by Tarantola
//! (1984) and Virieux & Operto (2009), with HU-to-acoustic properties supplied
//! by the provider-owned `SkullModel`.

#![expect(
    clippy::print_stdout,
    reason = "The example's console report is its user-visible output contract"
)]

#[path = "seismic_imaging/medium/mod.rs"]
mod seismic_medium;

#[path = "support/seismic_input.rs"]
mod seismic_input;

#[path = "transcranial_fwi/acquisition.rs"]
mod acquisition;
#[path = "transcranial_fwi/config.rs"]
mod config;
#[path = "transcranial_fwi/metrics.rs"]
mod metrics;
#[path = "transcranial_fwi/phantom.rs"]
mod phantom;
#[path = "transcranial_fwi/workflow.rs"]
mod workflow;

fn main() -> kwavers_core::error::KwaversResult<()> {
    workflow::run()
}
