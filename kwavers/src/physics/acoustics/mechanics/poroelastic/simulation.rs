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
    #[allow(dead_code)]
    grid: Grid,
    #[allow(dead_code)]
    pub material: PoroelasticMaterial,
    pub biot: BiotTheory,
}

impl PoroelasticSimulation {
    /// Create new poroelastic simulation
    pub fn new(grid: &Grid, material: PoroelasticMaterial) -> KwaversResult<Self> {
        let biot = BiotTheory::new(&material);

        Ok(Self {
            grid: grid.clone(),
            material,
            biot,
        })
    }

    /// Compute wave speeds at given frequency
    pub fn compute_wave_speeds(&self, frequency: f64) -> KwaversResult<WaveSpeeds> {
        self.biot.compute_wave_speeds(frequency)
    }

    /// Compute attenuation coefficients
    pub fn compute_attenuation(&self, frequency: f64) -> KwaversResult<(f64, f64)> {
        self.biot.compute_attenuation(frequency)
    }

    // Solver creation removed to decouple physics from solver layer.
    // Use crate::solver::forward::poroelastic::PoroelasticSolver directly.
}
