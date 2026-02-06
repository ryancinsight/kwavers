// signal/phase/mod.rs
use rand::Rng;
use std::fmt::Debug;

pub trait Phase: Debug + Send + Sync {
    fn phase(&self, t: f64) -> f64;
    fn clone_box(&self) -> Box<dyn Phase>;
}

impl Clone for Box<dyn Phase> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

#[derive(Debug, Clone)]
pub struct ConstantPhase {
    phase: f64,
}

impl ConstantPhase {
    #[must_use]
    pub fn new(phase: f64) -> Self {
        Self { phase }
    }
}

impl Phase for ConstantPhase {
    fn phase(&self, _t: f64) -> f64 {
        self.phase
    }
    fn clone_box(&self) -> Box<dyn Phase> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct RandomPhase {
    amplitude: f64,
}

impl RandomPhase {
    #[must_use]
    pub fn new(amplitude: f64) -> Self {
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");
        Self { amplitude }
    }
}

impl Phase for RandomPhase {
    fn phase(&self, _t: f64) -> f64 {
        let mut rng = rand::rngs::ThreadRng::default();
        rng.gen_range(-self.amplitude..=self.amplitude)
    }
    fn clone_box(&self) -> Box<dyn Phase> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct LinearPhaseShift {
    initial_phase: f64,
    rate: f64,
}

impl LinearPhaseShift {
    #[must_use]
    pub fn new(initial_phase: f64, rate: f64) -> Self {
        Self {
            initial_phase,
            rate,
        }
    }
}

impl Phase for LinearPhaseShift {
    fn phase(&self, t: f64) -> f64 {
        self.initial_phase + self.rate * t
    }
    fn clone_box(&self) -> Box<dyn Phase> {
        Box::new(self.clone())
    }
}
