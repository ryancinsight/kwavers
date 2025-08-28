// recorder/mod.rs - Main recorder module with modular structure

pub mod config;
pub mod detection;
pub mod events;
pub mod implementation;
pub mod statistics;
pub mod storage;
pub mod traits;

// Re-export key types for convenience
pub use config::RecorderConfig;
pub use events::{CavitationEvent, ThermalEvent};
pub use implementation::Recorder;
pub use statistics::RecorderStatistics;
pub use traits::RecorderTrait;
