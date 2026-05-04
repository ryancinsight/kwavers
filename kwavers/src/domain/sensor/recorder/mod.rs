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

pub use complex::Recorder;
pub use complex::Recorder as ComplexRecorder;
pub use config::{RecorderConfig, RecordingMode};
pub use fields::{SensorRecordField, SensorRecordSpec};
pub use pressure_statistics::{PressureFieldStatistics, SampledStatistics};
pub use simple::SensorRecorder as SimpleRecorder;
pub use statistics::RecorderStatistics;
pub use traits::RecorderTrait;
pub use velocity_statistics::{
    interpolate_staggered_to_collocated, SampledVelocityStats, VelocityComponentStats,
};
