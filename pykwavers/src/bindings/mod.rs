//! Narrow bindings sub-crate.
//!
//! Historically this module held a parallel refactor of the whole pykwavers
//! surface (grid/medium/source/sensor/simulation/array). That tree drifted
//! out of sync with the top-level definitions in `lib.rs` and is not wired
//! into the `_pykwavers` pymodule.
//!
//! Today only the photoacoustic high-level orchestration lives here, because
//! it depends on kwavers APIs that are not duplicated in `lib.rs`. The other
//! (broken, unused) submodules are intentionally not declared so they do not
//! gate compilation.
//!
//! Files `grid.rs`, `medium.rs`, `sensor.rs`, `simulation.rs`, `source.rs`,
//! `array.rs` remain on disk for reference but are dead code. Remove them if
//! you want to drop the legacy refactor entirely.

mod common;
mod photoacoustic;

pub use photoacoustic::{
    PhotoacousticOpticalModel, PhotoacousticRunResult, PhotoacousticRunner, PhotoacousticScenario,
};
