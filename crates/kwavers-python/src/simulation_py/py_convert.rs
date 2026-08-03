use kwavers_signal::Signal;
use leto::Array1;

/// Simple sine wave signal for testing
#[derive(Clone)]
pub(crate) struct SineSignal {
    pub(crate) frequency: f64,
    pub(crate) amplitude: f64,
}

impl SineSignal {
    pub(crate) fn new(frequency: f64, amplitude: f64) -> Self {
        Self {
            frequency,
            amplitude,
        }
    }
}

impl Signal for SineSignal {
    fn amplitude(&self, t: f64) -> f64 {
        self.amplitude * (2.0 * std::f64::consts::PI * self.frequency * t).sin()
    }

    fn duration(&self) -> Option<f64> {
        None // Continuous signal
    }

    fn frequency(&self, _t: f64) -> f64 {
        self.frequency
    }

    fn phase(&self, t: f64) -> f64 {
        2.0 * std::f64::consts::PI * self.frequency * t
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

impl std::fmt::Debug for SineSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SineSignal")
            .field("frequency", &self.frequency)
            .field("amplitude", &self.amplitude)
            .finish()
    }
}

/// Sampled signal from Python array
#[derive(Clone)]
pub(crate) struct SampledSignal {
    pub(crate) values: Array1<f64>,
    pub(crate) dt: f64,
}

impl SampledSignal {
    pub(crate) fn new(values: Array1<f64>, dt: f64) -> Self {
        Self { values, dt }
    }
}

impl Signal for SampledSignal {
    fn amplitude(&self, t: f64) -> f64 {
        if self.dt <= 0.0 || self.values.is_empty() {
            return 0.0;
        }
        let index = (t / self.dt).round() as isize;
        if index >= 0 && (index as usize) < self.values.len() {
            self.values[index as usize]
        } else {
            0.0
        }
    }

    fn duration(&self) -> Option<f64> {
        Some(self.values.len() as f64 * self.dt)
    }

    fn frequency(&self, _t: f64) -> f64 {
        0.0
    }

    fn phase(&self, _t: f64) -> f64 {
        0.0
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

impl std::fmt::Debug for SampledSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SampledSignal")
            .field("len", &self.values.len())
            .field("dt", &self.dt)
            .finish()
    }
}
