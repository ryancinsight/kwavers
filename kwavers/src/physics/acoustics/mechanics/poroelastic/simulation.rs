use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;

use super::biot::BiotTheory;
use super::material::PoroelasticMaterial;
use super::waves::WaveSpeeds;

/// Poroelastic simulation for wave propagation
///
/// # Example
///
/// ```no_run
/// use kwavers::physics::acoustics::mechanics::poroelastic::{
///     PoroelasticSimulation, PoroelasticMaterial
/// };
/// use kwavers::domain::grid::Grid;
///
/// # fn example() -> kwavers::core::error::KwaversResult<()> {
/// let grid = Grid::new(128, 128, 64, 1e-3, 1e-3, 1e-3)?;
/// let material = PoroelasticMaterial::from_tissue_type("trabecular_bone")?;
///
/// let sim = PoroelasticSimulation::new(&grid, material)?;
///
/// // Compute wave speeds
/// let speeds = sim.compute_wave_speeds(1e6)?;
/// println!("Fast wave: {} m/s", speeds.fast_wave);
/// println!("Slow wave: {} m/s", speeds.slow_wave);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct PoroelasticSimulation {
    pub material: PoroelasticMaterial,
    pub biot: BiotTheory,
}

impl PoroelasticSimulation {
    /// Create new poroelastic simulation
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(_grid: &Grid, material: PoroelasticMaterial) -> KwaversResult<Self> {
        let biot = BiotTheory::new(&material);

        Ok(Self { material, biot })
    }

    /// Compute wave speeds at given frequency
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn compute_wave_speeds(&self, frequency: f64) -> KwaversResult<WaveSpeeds> {
        self.biot.compute_wave_speeds(frequency)
    }

    /// Compute attenuation coefficients
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn compute_attenuation(&self, frequency: f64) -> KwaversResult<(f64, f64)> {
        self.biot.compute_attenuation(frequency)
    }

    // Solver creation removed to decouple physics from solver layer.
    // Use crate::solver::forward::poroelastic::PoroelasticSolver directly.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use crate::physics::acoustics::mechanics::poroelastic::PoroelasticMaterial;

    fn make_sim() -> PoroelasticSimulation {
        let grid = Grid::new(4, 4, 4, 0.001, 0.001, 0.001).unwrap();
        let material = PoroelasticMaterial::from_tissue_type("trabecular_bone").unwrap();
        PoroelasticSimulation::new(&grid, material).unwrap()
    }

    /// `new` constructs a simulation for a valid trabecular-bone material.
    #[test]
    fn new_succeeds_with_trabecular_bone() {
        let sim = make_sim();
        // Porosity of trabecular bone is 0.3
        assert!((sim.material.porosity - 0.3).abs() < 1e-10);
    }

    /// `compute_wave_speeds` delegates to BiotTheory and returns positive values.
    #[test]
    fn compute_wave_speeds_returns_positive_values() {
        let sim = make_sim();
        let speeds = sim.compute_wave_speeds(1e6).unwrap();
        assert!(speeds.fast_wave > 0.0, "fast_wave must be positive");
        assert!(speeds.slow_wave > 0.0, "slow_wave must be positive");
        assert!(speeds.shear_wave > 0.0, "shear_wave must be positive");
    }

    /// `compute_attenuation` returns two positive attenuation coefficients.
    #[test]
    fn compute_attenuation_returns_positive_coefficients() {
        let sim = make_sim();
        let (alpha_fast, alpha_slow) = sim.compute_attenuation(1e6).unwrap();
        assert!(alpha_fast > 0.0, "alpha_fast must be positive");
        assert!(alpha_slow > 0.0, "alpha_slow must be positive");
    }
}
