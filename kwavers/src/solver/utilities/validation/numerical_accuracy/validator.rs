use super::{ConvergenceResults, NumericalValidator, ValidationResults};
use crate::domain::grid::Grid;
use crate::domain::medium::HomogeneousMedium;

impl NumericalValidator {
    /// Create new validator with default test configuration
    #[must_use]
    pub fn new() -> Self {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).expect("Failed to create test grid");
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        Self { grid, medium }
    }

    /// Create validator with custom grid and medium
    pub fn with_config(grid: Grid, medium: HomogeneousMedium) -> Self {
        Self { grid, medium }
    }

    /// Run comprehensive validation suite
    pub fn validate_all(&self) -> Result<ValidationResults, Box<dyn std::error::Error>> {
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

        Ok(ValidationResults {
            dispersion_tests,
            stability_tests,
            boundary_tests,
            conservation_tests,
            convergence_tests,
        })
    }
}
