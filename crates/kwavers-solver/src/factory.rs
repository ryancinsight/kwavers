//! Solver selection policy for automated solver selection
//!
//! This module owns solver-selection policy only. Concrete solver construction
//! belongs to the simulation layer because construction binds domain objects to
//! numerical implementations.

use crate::config::SolverType;
use crate::interface::factory::{
    FactoryConfiguration, FactoryError, FactoryGridParameters, FactoryMediumParameters,
};

/// Solver selection policy evaluated over abstract problem descriptors.
#[derive(Debug)]
pub struct SolverFactoryRegistry;

impl SolverFactoryRegistry {
    /// Resolve `Auto` to a concrete solver type using the canonical policy.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn resolve_solver_type(
        solver_type: SolverType,
        grid: &dyn FactoryGridParameters,
        medium: &dyn FactoryMediumParameters,
    ) -> SolverType {
        if solver_type == SolverType::Auto {
            Self::select_best_solver(grid, medium)
        } else {
            solver_type
        }
    }

    /// Validate the solver workspace memory estimate against factory policy.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn validate_memory_budget(
        grid: &dyn FactoryGridParameters,
        config: &FactoryConfiguration,
    ) -> Result<usize, FactoryError> {
        let estimated_bytes = grid.total_points() * 8 * 4;
        if estimated_bytes > config.memory_budget {
            Err(FactoryError::ResourceExceeded {
                requested: estimated_bytes,
                available: config.memory_budget,
            })
        } else {
            Ok(estimated_bytes)
        }
    }

    /// Analyze the problem to select the best solver.
    pub fn select_best_solver(
        grid: &dyn FactoryGridParameters,
        medium: &dyn FactoryMediumParameters,
    ) -> SolverType {
        let is_heterogeneous = !medium.is_homogeneous();
        let large_grid = grid.total_points() > 1_000_000;

        if large_grid && !is_heterogeneous {
            SolverType::PSTD
        } else if is_heterogeneous {
            SolverType::Hybrid
        } else {
            SolverType::FDTD
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interface::factory::{FactoryGridParameters, FactoryMediumParameters};
    use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};

    struct TestGrid {
        nx: usize,
        ny: usize,
        nz: usize,
    }

    impl FactoryGridParameters for TestGrid {
        fn nx(&self) -> usize {
            self.nx
        }

        fn ny(&self) -> usize {
            self.ny
        }

        fn nz(&self) -> usize {
            self.nz
        }

        fn dx(&self) -> f64 {
            1.0e-3
        }

        fn dy(&self) -> f64 {
            1.0e-3
        }

        fn dz(&self) -> f64 {
            1.0e-3
        }
    }

    struct TestMedium {
        heterogeneity: f64,
    }

    impl FactoryMediumParameters for TestMedium {
        fn sound_speed(&self, _x: f64, _y: f64, _z: f64) -> f64 {
            SOUND_SPEED_WATER_SIM
        }

        fn density(&self, _x: f64, _y: f64, _z: f64) -> f64 {
            DENSITY_WATER_NOMINAL
        }

        fn heterogeneity(&self) -> f64 {
            self.heterogeneity
        }

        fn absorption(&self, _frequency: f64) -> f64 {
            0.0
        }
    }

    #[test]
    fn selects_fdtd_for_small_homogeneous_grid() {
        let grid = TestGrid {
            nx: 32,
            ny: 32,
            nz: 32,
        };
        let medium = TestMedium { heterogeneity: 0.0 };

        let selected = SolverFactoryRegistry::select_best_solver(&grid, &medium);

        assert_eq!(selected, SolverType::FDTD);
    }

    #[test]
    fn selects_pstd_for_large_homogeneous_grid() {
        let grid = TestGrid {
            nx: 128,
            ny: 128,
            nz: 128,
        };
        let medium = TestMedium { heterogeneity: 0.0 };

        let selected = SolverFactoryRegistry::select_best_solver(&grid, &medium);

        assert_eq!(selected, SolverType::PSTD);
    }

    #[test]
    fn selects_hybrid_for_heterogeneous_grid() {
        let grid = TestGrid {
            nx: 128,
            ny: 128,
            nz: 128,
        };
        let medium = TestMedium {
            heterogeneity: 0.01,
        };

        let selected = SolverFactoryRegistry::select_best_solver(&grid, &medium);

        assert_eq!(selected, SolverType::Hybrid);
    }

    #[test]
    fn reports_exact_memory_budget_violation() {
        let grid = TestGrid {
            nx: 4,
            ny: 5,
            nz: 6,
        };
        let config = FactoryConfiguration {
            memory_budget: 1024,
            ..Default::default()
        };

        let result = SolverFactoryRegistry::validate_memory_budget(&grid, &config);

        assert!(matches!(
            result,
            Err(FactoryError::ResourceExceeded {
                requested: 3840,
                available: 1024
            })
        ));
    }
}
