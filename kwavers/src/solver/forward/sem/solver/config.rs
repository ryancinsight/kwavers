//! `SemConfig` configuration type for the Spectral Element Method solver.

/// Configuration for SEM solver
#[derive(Debug, Clone)]
pub struct SemConfig {
    /// Polynomial degree for basis functions (2-8 recommended)
    pub polynomial_degree: usize,
    /// Wavenumber for Helmholtz equation (2πf/c)
    pub wavenumber: f64,
    /// Time step size
    pub dt: f64,
    /// Total number of time steps
    pub n_steps: usize,
    /// Sound speed (m/s)
    pub sound_speed: f64,
    /// Density (kg/m³)
    pub density: f64,
}

impl Default for SemConfig {
    fn default() -> Self {
        Self {
            polynomial_degree: 4,
            wavenumber: 1.0,
            dt: 1e-7,
            n_steps: 1000,
            sound_speed: 1500.0,
            density: 1000.0,
        }
    }
}
