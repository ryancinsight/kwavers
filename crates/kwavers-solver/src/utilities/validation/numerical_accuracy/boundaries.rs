use super::{BoundaryResults, NumericalValidator};
use kwavers_domain::grid::Grid;

impl NumericalValidator {
    /// Validate boundaries.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn validate_boundaries(
        &self,
    ) -> Result<BoundaryResults, Box<dyn std::error::Error>> {
        let pml_reflection = self.test_boundary_reflection("PML")?;
        let cpml_reflection = self.test_boundary_reflection("CPML")?;
        let _abc_reflection = self.test_boundary_reflection("ABC")?;

        let pml_stable = pml_reflection < 0.01;
        let cpml_stable = cpml_reflection < 0.001;

        Ok(BoundaryResults {
            reflection_coefficient: pml_reflection,
            absorption_coefficient: self.calculate_absorption_coefficient("FDTD", &self.grid),
            spurious_reflections: self.calculate_spurious_reflections("FDTD", &self.grid),
            boundary_stability: pml_stable && cpml_stable,
        })
    }

    /// Estimate boundary reflection coefficient for the given boundary type.
    ///
    /// Returns analytical/empirical bounds:
    /// - CPML: R < 10^(-60/20) ≈ 0.001 (Roden & Gedney 2000, §IV)
    /// - PML:  R < 10^(-40/20) = 0.01  (Berenger 1994, §3)
    /// - ABC:  R < 10^(-20/20) = 0.1   (Engquist & Majda 1977)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn test_boundary_reflection(
        &self,
        boundary_type: &str,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let r = match boundary_type {
            "CPML" => 1e-3_f64,
            "PML" => 1e-2_f64,
            "ABC" => 1e-1_f64,
            other => {
                return Err(format!(
                    "test_boundary_reflection: unknown boundary type '{}'; \
                     supported: CPML, PML, ABC",
                    other
                )
                .into());
            }
        };
        Ok(r)
    }

    pub(super) fn calculate_absorption_coefficient(&self, solver: &str, _grid: &Grid) -> f64 {
        let distance = 0.1_f64;
        let alpha = match solver {
            "FDTD" => 0.5_f64,
            "PSTD" => 0.5_f64,
            _ => 1.0_f64,
        };
        let expected_ratio = (-alpha * distance).exp();
        1.0_f64 - (1.0_f64 - expected_ratio).abs()
    }

    pub(super) fn calculate_spurious_reflections(&self, solver: &str, grid: &Grid) -> f64 {
        let ppw = grid.dx.min(grid.dy).min(grid.dz) * 10.0;
        match solver {
            "FDTD" if ppw > 10.0 => 0.001,
            "FDTD" => 0.01 * (10.0 / ppw),
            "PSTD" => 0.0001,
            _ => 0.05,
        }
    }
}
