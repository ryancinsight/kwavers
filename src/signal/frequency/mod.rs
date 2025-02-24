// signal/frequency/mod.rs

use std::fmt::Debug;

pub trait Frequency: Debug + Send + Sync {
    fn frequency(&self, t: f64) -> f64;
    fn clone_box(&self) -> Box<dyn Frequency>;
}

impl Clone for Box<dyn Frequency> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

#[derive(Debug, Clone)]
pub struct ConstantFrequency {
    value: f64,
}

impl ConstantFrequency {
    pub fn new(value: f64) -> Self {
        assert!(value > 0.0, "Frequency must be positive");
        Self { value }
    }
}

impl Frequency for ConstantFrequency {
    fn frequency(&self, _t: f64) -> f64 {
        self.value
    }
    fn clone_box(&self) -> Box<dyn Frequency> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct LinearSweep {
    start_freq: f64,
    end_freq: f64,
    duration: f64,
    rate: f64, // Sweep rate adjustment (default 1.0 for linear)
}

impl LinearSweep {
    pub fn new(start_freq: f64, end_freq: f64, duration: f64, rate: f64) -> Self {
        assert!(start_freq > 0.0 && end_freq > 0.0 && duration > 0.0 && rate > 0.0);
        Self {
            start_freq,
            end_freq,
            duration,
            rate,
        }
    }
}

impl Frequency for LinearSweep {
    fn frequency(&self, t: f64) -> f64 {
        if t >= self.duration {
            self.end_freq
        } else {
            self.start_freq
                + (self.end_freq - self.start_freq) * (t / self.duration).powf(self.rate)
        }
    }
    fn clone_box(&self) -> Box<dyn Frequency> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct QuadraticSweep {
    start_freq: f64,
    end_freq: f64,
    duration: f64,
    rate: f64, // Quadratic rate adjustment (default 2.0 for standard chirp)
}

impl QuadraticSweep {
    pub fn new(start_freq: f64, end_freq: f64, duration: f64, rate: f64) -> Self {
        assert!(start_freq > 0.0 && end_freq > 0.0 && duration > 0.0 && rate > 0.0);
        Self {
            start_freq,
            end_freq,
            duration,
            rate,
        }
    }
}

impl Frequency for QuadraticSweep {
    fn frequency(&self, t: f64) -> f64 {
        if t >= self.duration {
            self.end_freq
        } else {
            self.start_freq
                + (self.end_freq - self.start_freq) * (t / self.duration).powf(self.rate)
        }
    }
    fn clone_box(&self) -> Box<dyn Frequency> {
        Box::new(self.clone())
    }
}
