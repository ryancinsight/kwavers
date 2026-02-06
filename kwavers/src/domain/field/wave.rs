use ndarray::Array3;

/// Container for the primary acoustic wave fields
#[derive(Debug, Clone)]
pub struct WaveFields {
    /// Acoustic pressure
    pub p: Array3<f64>,
    /// Particle velocity x-component
    pub ux: Array3<f64>,
    /// Particle velocity y-component
    pub uy: Array3<f64>,
    /// Particle velocity z-component
    pub uz: Array3<f64>,
}

impl WaveFields {
    /// Create new zero-initialized wave fields with given shape
    pub fn new(shape: (usize, usize, usize)) -> Self {
        Self {
            p: Array3::zeros(shape),
            ux: Array3::zeros(shape),
            uy: Array3::zeros(shape),
            uz: Array3::zeros(shape),
        }
    }

    /// Get velocity components as a tuple
    pub fn velocity(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
        (&self.ux, &self.uy, &self.uz)
    }

    /// Get mutable velocity components as a tuple
    pub fn velocity_mut(&mut self) -> (&mut Array3<f64>, &mut Array3<f64>, &mut Array3<f64>) {
        (&mut self.ux, &mut self.uy, &mut self.uz)
    }
}
