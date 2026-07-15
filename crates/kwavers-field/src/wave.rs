use leto::Array3;

/// Container for the primary acoustic wave fields
#[derive(Debug, Clone)]
pub struct GenericWaveFields<T> {
    /// Acoustic pressure
    pub p: T,
    /// Particle velocity x-component
    pub ux: T,
    /// Particle velocity y-component
    pub uy: T,
    /// Particle velocity z-component
    pub uz: T,
}

pub type WaveFields = GenericWaveFields<Array3<f64>>;

impl WaveFields {
    /// Create new zero-initialized wave fields with given shape
    #[must_use]
    pub fn new(shape: (usize, usize, usize)) -> Self {
        let s = [shape.0, shape.1, shape.2];
        Self {
            p: Array3::<f64>::zeros(s),
            ux: Array3::<f64>::zeros(s),
            uy: Array3::<f64>::zeros(s),
            uz: Array3::<f64>::zeros(s),
        }
    }

    /// Get velocity components as a tuple
    #[must_use]
    pub fn velocity(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
        (&self.ux, &self.uy, &self.uz)
    }

    /// Get mutable velocity components as a tuple
    pub fn velocity_mut(&mut self) -> (&mut Array3<f64>, &mut Array3<f64>, &mut Array3<f64>) {
        (&mut self.ux, &mut self.uy, &mut self.uz)
    }
}
