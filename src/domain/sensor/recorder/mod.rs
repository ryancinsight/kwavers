// domain/sensor/recorder/mod.rs

pub mod complex;
pub mod config;
pub mod events;
pub mod simple;
pub mod statistics;
pub mod storage;
pub mod traits;

pub use complex::Recorder;
pub use complex::Recorder as ComplexRecorder;
pub use config::RecorderConfig;
pub use simple::SensorRecorder as SimpleRecorder;
pub use statistics::RecorderStatistics;
pub use traits::RecorderTrait;
