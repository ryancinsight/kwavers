//! Fixed-acquisition planning for repeated speed-shift frames.

mod batch;
mod construction;
mod prediction;
mod reconstruction;
#[cfg(test)]
mod tests;
mod types;
mod validation;
mod workspace;

pub use types::{
    SoundSpeedShiftBatch, SoundSpeedShiftBatchConfig, SoundSpeedShiftBatchFrame,
    SoundSpeedShiftFrameSummary, SoundSpeedShiftObjectiveHistoryPolicy, SoundSpeedShiftPlan,
};
