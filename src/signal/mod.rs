// signal/mod.rs

use std::fmt::Debug;

pub mod amplitude;
pub mod chirp;
pub mod frequency;
pub mod phase;
pub mod sine_wave;
pub mod sweep;

// Unused imports removed:
// use amplitude::Amplitude;
// use frequency::Frequency;
// use phase::Phase;

pub trait Signal: Debug + Send + Sync {
    fn amplitude(&self, t: f64) -> f64;
    fn duration(&self) -> Option<f64> {
        None
    }
    fn frequency(&self, t: f64) -> f64;
    fn phase(&self, t: f64) -> f64;
    fn clone_box(&self) -> Box<dyn Signal>;
}

impl Clone for Box<dyn Signal> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

pub use chirp::ChirpSignal;
pub use sine_wave::SineWave;
pub use sweep::SweepSignal;
