// domain/sensor/recorder/mod.rs

pub mod complex;
pub mod config;
pub mod events;
pub mod fields;
pub mod pressure_statistics;
pub mod simple;
pub mod statistics;
pub mod storage;
pub mod traits;
pub mod velocity_statistics;

pub(super) const STATISTICS_CHUNK_SIZE: usize = 4096;

pub use complex::Recorder;
pub use config::{RecorderConfig, RecordingMode};
pub use fields::{SensorRecordField, SensorRecordSpec};
pub use pressure_statistics::{PressureFieldStatistics, SampledStatistics};
pub use simple::SensorRecorder;
pub use statistics::RecorderStatistics;
pub use traits::RecorderTrait;
pub use velocity_statistics::{
    interpolate_staggered_to_collocated, SampledVelocityStats, VelocityComponentStats,
};
