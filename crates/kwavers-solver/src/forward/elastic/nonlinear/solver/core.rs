use super::super::config::NonlinearSWEConfig;
use super::super::material::HyperelasticModel;
use super::super::numerics::NumericsOperators;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::Medium;

/// Nonlinear elastic wave equation solver
#[derive(Debug)]
pub struct NonlinearElasticWaveSolver {
    pub(super) grid: Grid,
    pub(super) _material: HyperelasticModel,
    pub(super) config: NonlinearSWEConfig,
    pub(super) attenuation_np_per_m: f64,
    pub(super) numerics: NumericsOperators,
}

impl NonlinearElasticWaveSolver {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(
        grid: &Grid,
        medium: &dyn Medium,
        material: HyperelasticModel,
        config: NonlinearSWEConfig,
    ) -> KwaversResult<Self> {
        let attenuation_np_per_m = medium
            .optical_absorption_coefficient(0.0, 0.0, 0.0, grid)
            .max(0.0);

        let numerics = NumericsOperators::new(grid.clone());

        Ok(Self {
            grid: grid.clone(),
            _material: material,
            config,
            attenuation_np_per_m,
            numerics,
        })
    }
}
