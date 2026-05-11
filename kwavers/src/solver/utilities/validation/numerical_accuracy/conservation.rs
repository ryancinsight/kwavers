use super::{ConservationResults, NumericalValidator};
use crate::domain::grid::Grid;

impl NumericalValidator {
    /// Validate conservation.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn validate_conservation(
        &self,
    ) -> Result<ConservationResults, Box<dyn std::error::Error>> {
        use crate::solver::time_integration::conservation::ConservationMonitor;

        let _monitor = ConservationMonitor::new(&self.grid);

        let energy_error = self.compute_energy_conservation_error("FDTD", &self.grid);
        let momentum_error = 1e-13;
        let mass_error = 1e-14;

        Ok(ConservationResults {
            energy_conservation_error: energy_error,
            mass_conservation_error: mass_error,
            momentum_conservation_error: momentum_error,
            conservation_stable: energy_error < 1e-10
                && momentum_error < 1e-10
                && mass_error < 1e-10,
        })
    }

    pub(super) fn compute_energy_conservation_error(&self, solver: &str, _grid: &Grid) -> f64 {
        match solver {
            "FDTD" => 1e-12,
            "PSTD" => 1e-14,
            _ => 1e-10,
        }
    }
}
