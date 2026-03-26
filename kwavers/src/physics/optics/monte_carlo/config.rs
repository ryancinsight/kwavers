/// Monte Carlo simulation configuration
#[derive(Clone, Debug)]
pub struct SimulationConfig {
    pub num_photons: usize,
    pub max_steps: usize,
    pub russian_roulette_threshold: f64,
    pub russian_roulette_survival: f64,
    pub boundary_reflection: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            num_photons: 100_000,
            max_steps: 10_000,
            russian_roulette_threshold: 0.001,
            russian_roulette_survival: 0.1,
            boundary_reflection: false,
        }
    }
}

impl SimulationConfig {
    /// Set number of photons
    pub fn num_photons(mut self, n: usize) -> Self {
        self.num_photons = n;
        self
    }

    /// Set maximum steps per photon
    pub fn max_steps(mut self, n: usize) -> Self {
        self.max_steps = n;
        self
    }

    /// Set Russian roulette threshold
    pub fn russian_roulette_threshold(mut self, threshold: f64) -> Self {
        self.russian_roulette_threshold = threshold;
        self
    }

    /// Enable/disable boundary reflection
    pub fn boundary_reflection(mut self, enabled: bool) -> Self {
        self.boundary_reflection = enabled;
        self
    }
}
