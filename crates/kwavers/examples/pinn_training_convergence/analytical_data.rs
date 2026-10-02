//! Analytical plane-wave solution and training-data generation for the convergence study.

/// Training parameters for PINN experiments
#[derive(Debug, Clone)]
pub(crate) struct ExperimentConfig {
    /// Number of spatial points (N×N grid)
    pub(crate) num_points: usize,
    /// Number of epochs
    pub(crate) epochs: usize,
    /// Learning rate
    pub(crate) learning_rate: f64,
    /// Network hidden layer sizes
    pub(crate) hidden_layers: Vec<usize>,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            num_points: 32,
            epochs: 1000,
            learning_rate: 1e-3,
            hidden_layers: vec![64, 64, 64, 64],
        }
    }
}

/// Analytical solution for plane wave
#[derive(Debug, Clone)]
pub(crate) struct PlaneWaveAnalytical {
    pub(crate) amplitude: f64,
    pub(crate) wave_number: f64,
    pub(crate) omega: f64,
    pub(crate) direction: [f64; 2],
}

impl PlaneWaveAnalytical {
    /// Create P-wave plane wave solution
    pub(crate) fn new(amplitude: f64, wavelength: f64, c_p: f64) -> Self {
        let wave_number = 2.0 * std::f64::consts::PI / wavelength;
        let omega = c_p * wave_number;
        Self {
            amplitude,
            wave_number,
            omega,
            direction: [1.0, 0.0], // Propagating in +x direction
        }
    }

    /// Evaluate displacement at (x, y, t)
    fn displacement(&self, x: f64, y: f64, t: f64) -> [f64; 2] {
        let phase =
            self.wave_number * (self.direction[0] * x + self.direction[1] * y) - self.omega * t;
        let u = self.amplitude * phase.sin();
        [u * self.direction[0], u * self.direction[1]]
    }
}

/// Generate training data from analytical solution
pub(crate) fn generate_training_data(
    solution: &PlaneWaveAnalytical,
    num_points: usize,
    domain_size: f64,
    t_max: f64,
) -> (Vec<[f64; 3]>, Vec<[f64; 2]>) {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    let dx = domain_size / (num_points as f64);
    let dt = t_max / 10.0; // Sample 10 time steps

    for ti in 0..10 {
        let t = ti as f64 * dt;
        for i in 0..num_points {
            for j in 0..num_points {
                let x = i as f64 * dx;
                let y = j as f64 * dx;

                inputs.push([t, x, y]);
                targets.push(solution.displacement(x, y, t));
            }
        }
    }

    (inputs, targets)
}
