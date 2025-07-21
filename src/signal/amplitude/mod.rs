// signal/amplitude/mod.rs

use std::fmt::Debug;

pub trait Amplitude: Debug + Send + Sync {
    fn amplitude(&self, t: f64) -> f64;
    fn clone_box(&self) -> Box<dyn Amplitude>;
}

impl Clone for Box<dyn Amplitude> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

#[derive(Debug, Clone)]
pub struct ConstantAmplitude {
    value: f64,
}

impl ConstantAmplitude {
    pub fn new(value: f64) -> Self {
        assert!(value >= 0.0, "Amplitude must be non-negative");
        Self { value }
    }
}

impl Amplitude for ConstantAmplitude {
    fn amplitude(&self, _t: f64) -> f64 {
        self.value
    }
    fn clone_box(&self) -> Box<dyn Amplitude> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct PowerModulation {
    base_amplitude: f64,
    modulation_freq: f64,  // Frequency of amplitude modulation (Hz)
    modulation_depth: f64, // Depth of modulation (0 to 1)
}

impl PowerModulation {
    pub fn new(base_amplitude: f64, modulation_freq: f64, modulation_depth: f64) -> Self {
        assert!(
            base_amplitude >= 0.0
                && modulation_freq > 0.0
                && (0.0..=1.0).contains(&modulation_depth)
        );
        Self {
            base_amplitude,
            modulation_freq,
            modulation_depth,
        }
    }
}

impl Amplitude for PowerModulation {
    fn amplitude(&self, t: f64) -> f64 {
        self.base_amplitude
            * (1.0
                + self.modulation_depth
                    * (2.0 * std::f64::consts::PI * self.modulation_freq * t).sin())
    }
    fn clone_box(&self) -> Box<dyn Amplitude> {
        Box::new(self.clone())
    }
}
