use super::config::FdtdFemCouplingConfig;
use super::interface::FdtdFemInterface;
use crate::forward::helmholtz::fem::solver::{FemHelmholtzConfig, FemHelmholtzSolver};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_mesh::tetrahedral::TetrahedralMesh;
use ndarray::Array3;
use kwavers_math::fft::Complex64;

/// FDTD-FEM Schwarz Coupler
#[derive(Debug)]
pub struct FdtdFemCoupler {
    pub(super) config: FdtdFemCouplingConfig,
    pub(super) interface: FdtdFemInterface,
    fem_solver: FemHelmholtzSolver,
    convergence_history: Vec<f64>,
}

impl FdtdFemCoupler {
    /// Create new FDTD-FEM coupler
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(
        config: FdtdFemCouplingConfig,
        fdtd_grid: &Grid,
        fem_mesh: &TetrahedralMesh,
    ) -> KwaversResult<Self> {
        let interface = FdtdFemInterface::new(fdtd_grid, fem_mesh)?;

        let fem_solver = FemHelmholtzSolver::new(FemHelmholtzConfig::default(), fem_mesh.clone());

        Ok(Self {
            config,
            interface,
            fem_solver,
            convergence_history: Vec::new(),
        })
    }

    /// Perform Schwarz iteration between FDTD and FEM domains
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn schwarz_iteration(
        &mut self,
        fdtd_field: &mut Array3<f64>,
        fem_field: &mut Vec<f64>,
        fdtd_grid: &Grid,
        fem_mesh: &TetrahedralMesh,
        iteration: usize,
    ) -> KwaversResult<f64> {
        let fdtd_interface_values = self.extract_fdtd_interface(fdtd_field)?;
        self.update_fem_boundary(fem_field.as_mut_slice(), &fdtd_interface_values, fem_mesh)?;
        self.solve_fem_domain(fem_field, fem_mesh, fdtd_grid)?;
        let fem_interface_values = self.extract_fem_interface(fem_field.as_slice())?;
        let residual =
            self.update_fdtd_boundary(fdtd_field, &fem_interface_values, fdtd_grid, iteration)?;
        self.convergence_history.push(residual);
        Ok(residual)
    }

    /// Solve FEM domain using coupled values
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn solve_fem_domain(
        &mut self,
        fem_field: &mut [f64],
        _fem_mesh: &TetrahedralMesh,
        grid: &Grid,
    ) -> KwaversResult<()> {
        self.fem_solver.boundary_manager().clear();

        let mut dirichlet_bcs = Vec::new();
        for &fem_idx in &self.interface.fem_indices {
            if fem_idx < fem_field.len() {
                let value = Complex64::new(fem_field[fem_idx], 0.0);
                dirichlet_bcs.push((fem_idx, value));
            }
        }
        self.fem_solver
            .boundary_manager()
            .add_dirichlet(dirichlet_bcs);

        let medium = HomogeneousMedium::water(grid);
        self.fem_solver.assemble_system(&medium)?;
        self.fem_solver.solve_system()?;

        let solution = self.fem_solver.solution();
        for (i, val) in solution.iter().enumerate() {
            if i < fem_field.len() {
                fem_field[i] = val.re;
            }
        }

        Ok(())
    }

    /// Extract FDTD field values at interface
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn extract_fdtd_interface(&self, fdtd_field: &Array3<f64>) -> KwaversResult<Vec<f64>> {
        let mut interface_values = Vec::with_capacity(self.interface.fdtd_indices.len());
        for &(i, j, k) in &self.interface.fdtd_indices {
            interface_values.push(fdtd_field[[i, j, k]]);
        }
        Ok(interface_values)
    }

    /// Update FEM boundary conditions with FDTD interface values
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn update_fem_boundary(
        &self,
        fem_field: &mut [f64],
        fdtd_values: &[f64],
        _fem_mesh: &TetrahedralMesh,
    ) -> KwaversResult<()> {
        for (&fem_idx, &fdtd_value) in self.interface.fem_indices.iter().zip(fdtd_values.iter()) {
            if fem_idx < fem_field.len() {
                let current_value = fem_field[fem_idx];
                fem_field[fem_idx] = self.config.relaxation_factor.mul_add(
                    fdtd_value,
                    (1.0 - self.config.relaxation_factor) * current_value,
                );
            }
        }
        Ok(())
    }

    /// Extract FEM field values at interface
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn extract_fem_interface(&self, fem_field: &[f64]) -> KwaversResult<Vec<f64>> {
        let mut interface_values = Vec::with_capacity(self.interface.fem_indices.len());
        for &fem_idx in &self.interface.fem_indices {
            let value = fem_field.get(fem_idx).ok_or_else(|| {
                kwavers_core::error::KwaversError::InvalidInput(format!(
                    "FEM interface node index {} is out of bounds (fem_field len {})",
                    fem_idx,
                    fem_field.len()
                ))
            })?;
            interface_values.push(*value);
        }
        Ok(interface_values)
    }

    /// Update FDTD boundary conditions with FEM interface values
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn update_fdtd_boundary(
        &self,
        fdtd_field: &mut Array3<f64>,
        fem_values: &[f64],
        fdtd_grid: &Grid,
        iteration: usize,
    ) -> KwaversResult<f64> {
        let mut max_residual = 0.0;

        for (&(i, j, k), &fem_value) in self.interface.fdtd_indices.iter().zip(fem_values.iter()) {
            let current_value = fdtd_field[[i, j, k]];
            let new_value = self.config.relaxation_factor.mul_add(
                fem_value,
                (1.0 - self.config.relaxation_factor) * current_value,
            );

            let residual = (new_value - current_value).abs();
            if residual > max_residual {
                max_residual = residual;
            }

            fdtd_field[[i, j, k]] = new_value;
        }

        if iteration > 0 {
            self.apply_interface_smoothing(fdtd_field, fdtd_grid)?;
        }

        Ok(max_residual)
    }

    /// Apply smoothing to interface region for stability
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn apply_interface_smoothing(&self, field: &mut Array3<f64>, grid: &Grid) -> KwaversResult<()> {
        for &(i, j, k) in &self.interface.fdtd_indices {
            if i > 0 && i < grid.nx - 1 && j > 0 && j < grid.ny - 1 && k > 0 && k < grid.nz - 1 {
                let laplacian = 6.0f64.mul_add(
                    -field[[i, j, k]],
                    field[[i - 1, j, k]]
                        + field[[i + 1, j, k]]
                        + field[[i, j - 1, k]]
                        + field[[i, j + 1, k]]
                        + field[[i, j, k - 1]]
                        + field[[i, j, k + 1]],
                );
                field[[i, j, k]] += 0.1 * laplacian;
            }
        }
        Ok(())
    }

    /// Check convergence of Schwarz iteration
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[must_use]
    pub fn check_convergence(&self) -> bool {
        if self.convergence_history.len() < 2 {
            return false;
        }
        let recent_residual = *self.convergence_history.last().unwrap();
        recent_residual < self.config.tolerance
    }

    /// Get convergence history
    #[must_use]
    pub fn convergence_history(&self) -> &[f64] {
        &self.convergence_history
    }

    /// Reset convergence tracking
    pub fn reset_convergence(&mut self) {
        self.convergence_history.clear();
    }
}
