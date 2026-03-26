use crate::domain::grid::GridDimensions;
use std::fmt;

/// Monte Carlo simulation result
#[derive(Debug)]
pub struct MCResult {
    pub(crate) dimensions: GridDimensions,
    pub(crate) absorbed_energy: Vec<f64>,
    pub(crate) fluence: Vec<f64>,
    pub(crate) num_photons: usize,
}

impl MCResult {
    /// Get absorbed energy map (J/m³)
    pub fn absorbed_energy(&self) -> &[f64] {
        &self.absorbed_energy
    }

    /// Get fluence map (J/m²)
    pub fn fluence(&self) -> &[f64] {
        &self.fluence
    }

    /// Get total absorbed energy (J)
    pub fn total_absorbed_energy(&self) -> f64 {
        self.absorbed_energy.iter().sum()
    }

    /// Get fluence normalized by number of photons
    pub fn normalized_fluence(&self) -> Vec<f64> {
        let norm = 1.0 / self.num_photons as f64;
        self.fluence.iter().map(|&f| f * norm).collect()
    }

    /// Get dimensions
    pub fn dimensions(&self) -> GridDimensions {
        self.dimensions
    }
}

impl fmt::Display for MCResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total_absorbed = self.total_absorbed_energy();
        let mean_fluence = self.fluence.iter().sum::<f64>() / self.fluence.len() as f64;
        write!(
            f,
            "MCResult(photons={}, absorbed={:.3e} J, mean_fluence={:.3e} J/m²)",
            self.num_photons, total_absorbed, mean_fluence
        )
    }
}
