//! `solve_coupled` — iterative BEM-FEM coupling loop.

use num_complex::Complex64;

use kwavers_core::error::KwaversResult;
use kwavers_domain::mesh::tetrahedral::TetrahedralMesh;

use super::BemFemCoupler;

impl BemFemCoupler {
    /// Perform the coupled BEM-FEM solution via Dirichlet-Neumann iteration.
    ///
    /// Each iteration:
    /// 1. Extract FEM field at the coupling interface.
    /// 2. Transfer FEM values to the BEM boundary (with relaxation).
    /// 3. Solve the BEM system (rigid CFIE).
    /// 4. Transfer BEM solution to the FEM boundary; compute residual.
    /// 5. Solve the FEM linear system.
    ///
    /// Stops when `residual < config.convergence_tolerance` or
    /// `config.max_iterations` is reached.
    ///
    /// # Errors
    /// Propagates errors from all sub-solve steps.
    pub fn solve_coupled(
        &mut self,
        fem_field: &mut Vec<Complex64>,
        bem_boundary_values: &mut Vec<Complex64>,
        fem_mesh: &TetrahedralMesh,
        wavenumber: f64,
    ) -> KwaversResult<f64> {
        self.convergence_history.clear();
        let mut residual = f64::INFINITY;

        let fem_matrix = self.assemble_system_matrix(fem_mesh, wavenumber)?;

        for iteration in 0..self.config.max_iterations {
            let fem_interface_values = self.extract_fem_interface(fem_field.as_slice())?;
            self.apply_to_bem_boundary(&fem_interface_values, bem_boundary_values.as_mut_slice())?;
            self.solve_bem_system(bem_boundary_values.as_mut_slice(), wavenumber)?;
            let bem_interface_values =
                self.extract_bem_interface(bem_boundary_values.as_slice())?;
            residual = self.apply_to_fem_boundary(
                &bem_interface_values,
                fem_field.as_mut_slice(),
                fem_mesh,
            )?;
            self.solve_linear_system(&fem_matrix, fem_field.as_mut_slice())?;

            self.convergence_history.push(residual);
            self.iteration_count = iteration + 1;

            if residual < self.config.convergence_tolerance {
                break;
            }
        }

        Ok(residual)
    }
}
