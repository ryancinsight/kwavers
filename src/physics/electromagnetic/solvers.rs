//! Electromagnetic Physics Solvers
//!
//! This module provides physics-specific solver implementations that combine
//! electromagnetic constitutive relations with numerical algorithms from the
//! shared solver layer.

use crate::core::error::KwaversResult;
use crate::domain::field::EMFields;
use crate::domain::grid::Grid;
use crate::physics::electromagnetic::equations::{
    EMDimension, EMMaterialDistribution, ElectromagneticWaveEquation,
};
use crate::solver::forward::fdtd::ElectromagneticFdtdSolver;

/// Unified electromagnetic solver interface
///
/// This enum provides a unified interface to different electromagnetic
/// solvers while maintaining physics-specific behavior.
#[derive(Debug)]
pub enum ElectromagneticSolver {
    /// Finite Difference Time Domain solver for Maxwell's equations
    FDTD(ElectromagneticFdtdSolver),
    // Future: FEM, SEM, etc.
}

impl ElectromagneticSolver {
    /// Create an FDTD electromagnetic solver
    pub fn fdtd(grid: Grid, materials: EMMaterialDistribution, dt: f64) -> KwaversResult<Self> {
        let solver = ElectromagneticFdtdSolver::new(grid, materials, dt, 4)?;
        Ok(Self::FDTD(solver))
    }
}

impl ElectromagneticWaveEquation for ElectromagneticSolver {
    fn em_dimension(&self) -> EMDimension {
        match self {
            ElectromagneticSolver::FDTD(fdtd) => fdtd.em_dimension(),
        }
    }

    fn material_properties(&self) -> &EMMaterialDistribution {
        match self {
            ElectromagneticSolver::FDTD(fdtd) => fdtd.material_properties(),
        }
    }

    fn em_fields(&self) -> &EMFields {
        match self {
            ElectromagneticSolver::FDTD(fdtd) => fdtd.em_fields(),
        }
    }

    fn step_maxwell(&mut self, dt: f64) -> Result<(), String> {
        match self {
            ElectromagneticSolver::FDTD(fdtd) => fdtd.step_maxwell(dt),
        }
    }

    fn apply_em_boundary_conditions(&mut self, fields: &mut EMFields) {
        match self {
            ElectromagneticSolver::FDTD(fdtd) => fdtd.apply_em_boundary_conditions(fields),
        }
    }

    fn check_em_constraints(&self, fields: &EMFields) -> Result<(), String> {
        match self {
            ElectromagneticSolver::FDTD(fdtd) => fdtd.check_em_constraints(fields),
        }
    }
}

/// Convenience type alias for FDTD electromagnetic solver
pub type FdtdElectromagneticSolver = ElectromagneticFdtdSolver;

/// Placeholder for future FEM electromagnetic solver
/// This would adapt the existing Helmholtz FEM solver for electromagnetic problems
#[derive(Debug)]
pub struct FemElectromagneticSolver {
    // Future implementation
}

impl FemElectromagneticSolver {
    pub fn new() -> Self {
        Self {}
    }
}

// Placeholder implementation
impl ElectromagneticWaveEquation for FemElectromagneticSolver {
    fn em_dimension(&self) -> EMDimension {
        EMDimension::Three
    }

    fn material_properties(&self) -> &EMMaterialDistribution {
        static MATERIALS: std::sync::OnceLock<EMMaterialDistribution> = std::sync::OnceLock::new();
        MATERIALS.get_or_init(|| EMMaterialDistribution::vacuum(&[1]))
    }

    fn em_fields(&self) -> &EMFields {
        static FIELDS: std::sync::OnceLock<EMFields> = std::sync::OnceLock::new();
        FIELDS.get_or_init(|| EMFields {
            electric: ndarray::ArrayD::zeros(ndarray::IxDyn(&[3])),
            magnetic: ndarray::ArrayD::zeros(ndarray::IxDyn(&[3])),
            displacement: None,
            flux_density: None,
        })
    }

    fn step_maxwell(&mut self, _dt: f64) -> Result<(), String> {
        // Future: implement FEM time stepping for Maxwell's equations
        Ok(())
    }

    fn apply_em_boundary_conditions(&mut self, _fields: &mut EMFields) {
        // Future: implement FEM boundary conditions
    }

    fn check_em_constraints(&self, _fields: &EMFields) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;

    #[test]
    fn test_electromagnetic_solver_creation() {
        let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();

        // Use canonical domain composition pattern
        let materials = EMMaterialDistribution::vacuum(&[10, 10, 10]);

        let solver = ElectromagneticSolver::fdtd(grid, materials, 1e-12).unwrap();
        assert_eq!(solver.em_dimension(), EMDimension::Three);
    }
}
