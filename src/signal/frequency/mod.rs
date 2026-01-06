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
    #[must_use]
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
