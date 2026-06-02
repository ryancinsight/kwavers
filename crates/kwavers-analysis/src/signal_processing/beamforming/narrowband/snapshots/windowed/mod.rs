#![deny(missing_docs)]
//! Windowed snapshot extraction for narrowband adaptive array processing.

mod extraction;
mod types;

#[cfg(test)]
mod tests;

pub use extraction::{extract_stft_bin_snapshots, extract_windowed_snapshots};
pub use types::{
    SnapshotMethod, SnapshotScenario, SnapshotSelection, StftBinConfig, WindowFunction,
};
