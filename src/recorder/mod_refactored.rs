// recorder/mod.rs - Main recorder module (refactored)

pub mod config;
pub mod detection;
pub mod events;
pub mod implementation;
pub mod statistics;
pub mod storage;
pub mod traits;

// Re-export key types for convenience
pub use config::RecorderConfig;
pub use events::{CavitationEvent, EventCollection, ThermalEvent};
pub use implementation::Recorder;
pub use statistics::RecorderStatistics;
pub use traits::RecorderTrait;

// Legacy compatibility exports
pub use implementation::Recorder as MainRecorder;