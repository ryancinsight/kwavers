//! Sensor-Specific Beamforming Interface
//!
//! Provides domain-layer beamforming operations tightly coupled to sensor array
//! geometries and hardware characteristics. Serves as the accessor layer between
//! domain sensor concerns and analysis-layer algorithms.
//!
//! | Submodule    | Contents                                              |
//! |--------------|-------------------------------------------------------|
//! | `types`      | `BeamformerWindowType` enum, `SensorProcessingParams` struct   |
//! | `beamformer` | `SensorBeamformer` — delay, windowing, steering       |

mod beamformer;
#[cfg(test)]
mod tests;
mod types;

pub use beamformer::SensorBeamformer;
pub use types::{BeamformerWindowType, SensorProcessingParams};
