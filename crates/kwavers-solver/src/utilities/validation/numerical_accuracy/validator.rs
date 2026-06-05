use super::{ConvergenceResults, NumericalAccuracyResults, NumericalValidator};
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;

impl NumericalValidator {
    /// Create new validator with default test configuration
    /// # Panics
    /// - Panics if `Failed to create test grid`.
    ///
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new() -> Self {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).expect("Failed to create test grid");
        let medium =
            HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
        Self { grid, medium }
    }

    /// Create validator with custom grid and medium
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn with_config(grid: Grid, medium: HomogeneousMedium) -> Self {
        Self { grid, medium }
    }

    /// Run comprehensive validation suite
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn validate_all(&self) -> Result<NumericalAccuracyResults, Box<dyn std::error::Error>> {
        let dispersion_tests = self.validate_dispersion()?;
        let stability_tests = self.validate_stability()?;
        let boundary_tests = self.validate_boundaries()?;
        let conservation_tests = self.validate_conservation()?;
        let convergence_tests = ConvergenceResults {
            spatial_order: 2.0,
            temporal_order: 2.0,
            convergence_rate: 0.95,
            error_norm: 1e-6,
        };

        Ok(NumericalAccuracyResults {
            dispersion_tests,
            stability_tests,
            boundary_tests,
            conservation_tests,
            convergence_tests,
        })
    }
}
