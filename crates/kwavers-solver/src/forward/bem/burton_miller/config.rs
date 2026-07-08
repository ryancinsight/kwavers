use kwavers_core::constants::numerical::TWO_PI;
use kwavers_math::fft::Complex64;

/// Configuration for Burton-Miller BEM formulation
#[derive(Debug, Clone, Copy)]
pub struct BurtonMillerConfig {
    pub wavenumber: f64,
    /// Optimal coupling: α = 1/(ik) = -i/k
    pub coupling_alpha: Complex64,
    pub frequency: f64,
    pub sound_speed: f64,
    pub singular_regularization: f64,
    pub assembly_tolerance: f64,
}

impl BurtonMillerConfig {
    #[must_use]
    pub fn new(frequency: f64, sound_speed: f64) -> Self {
        let wavenumber = TWO_PI * frequency / sound_speed;
        let coupling_alpha = Complex64::new(0.0, -1.0 / wavenumber);
        Self {
            wavenumber,
            coupling_alpha,
            frequency,
            sound_speed,
            singular_regularization: 1e-10,
            assembly_tolerance: 1e-12,
        }
    }

    #[must_use]
    pub fn with_coupling_alpha(mut self, alpha: Complex64) -> Self {
        self.coupling_alpha = alpha;
        self
    }
}
