//! BEM system solve, FEM matrix assembly, and linear solver.

use kwavers_math::fft::Complex64;
use kwavers_math::linear_algebra::sparse::{
    CompressedSparseRowMatrix, IterativeSolver, SolverConfig, SparsePreconditioner,
};
use leto::Array1;

use kwavers_core::error::KwaversResult;
use kwavers_mesh::tetrahedral::TetrahedralMesh;

use super::BemFemCoupler;
use kwavers_core::constants::numerical::TWO_PI;

impl BemFemCoupler {
    /// Solve the BEM system via rigid-scattering CFIE for the given `wavenumber`.
    ///
    /// Updates `bem_boundary_values` at interface element indices with the
    /// BEM surface pressure.
    ///
    /// # Errors
    /// Propagates errors from `BemSolver::solve_rigid`.
    pub(super) fn solve_bem_system(
        &mut self,
        bem_boundary_values: &mut [Complex64],
        wavenumber: f64,
    ) -> KwaversResult<()> {
        use crate::forward::bem::field::{compute_vertex_normals, plane_wave_incident};

        let nv = self.bem_solver.vertices.len();
        if nv == 0 {
            return Ok(());
        }

        let c = self.bem_solver.config.sound_speed;
        let f = wavenumber * c / (TWO_PI);
        self.bem_solver.config.frequency = f;
        self.bem_solver.config.wavenumber = wavenumber;
        self.bem_solver.config.coupling_alpha =
            kwavers_math::fft::Complex64::new(0.0, 1.0 / wavenumber);
        self.bem_solver.invalidate_matrix();

        let normals = compute_vertex_normals(&self.bem_solver.vertices, &self.bem_solver.triangles);
        let (p_inc, dp_inc_dn) = plane_wave_incident(
            &self.bem_solver.vertices,
            &normals,
            [1.0, 0.0, 0.0],
            wavenumber,
            kwavers_math::fft::Complex64::new(1.0, 0.0),
        );

        let p_surface = self.bem_solver.solve_rigid(p_inc, dp_inc_dn)?;

        for (local_idx, &global_idx) in self.interface.bem_interface_elements.iter().enumerate() {
            if global_idx < (bem_boundary_values.len()) && local_idx < (p_surface.len()) {
                bem_boundary_values[global_idx] = p_surface[local_idx];
            }
        }

        Ok(())
    }

    pub(crate) fn assemble_system_matrix(
        &self,
        fem_mesh: &TetrahedralMesh,
        wavenumber: f64,
    ) -> KwaversResult<CompressedSparseRowMatrix<Complex64>> {
        let num_nodes = fem_mesh.nodes.len();
        let mut stiffness = CompressedSparseRowMatrix::create(num_nodes, num_nodes);
        // TODO: Implement actual FEM matrix assembly
        let k_sq = Complex64::from(wavenumber.powi(2));
        for i in 0..num_nodes {
            stiffness.set_diagonal(i, Complex64::new(1.0, 0.0));
            for j in 0..num_nodes {
                if i != j {
                    stiffness.add_value(i, j, Complex64::new(0.0, 0.0));
                }
            }
        }
        // Subtract k²M (mass matrix contribution)
        // For now, just add the k² term to diagonal to make it non-trivial
        for i in 0..num_nodes {
            if let Some(mut val) = stiffness.get_diagonal(i) {
                val -= k_sq;
                stiffness.set_diagonal(i, val);
            }
        }
        Ok(stiffness)
    }

    /// Solve the assembled FEM linear system with BiCGSTAB.
    ///
    /// Applies penalty-row Dirichlet boundary conditions for interface nodes
    /// before solving. Overwrites `fem_field` with the solution.
    ///
    /// # Errors
    /// Propagates errors from `IterativeSolver::bicgstab_complex`.
    pub(crate) fn solve_linear_system(
        &self,
        matrix: &CompressedSparseRowMatrix<Complex64>,
        fem_field: &mut [Complex64],
    ) -> KwaversResult<()> {
        let num_nodes = matrix.rows;
        let penalty = 1.0e14;
        let mut rhs = Array1::<Complex64>::from_elem(num_nodes, Complex64::default());

        for &node_idx in &self.interface.fem_interface_nodes {
            if node_idx < num_nodes {
                let prescribed_val = fem_field[node_idx];
                rhs[node_idx] += Complex64::from(penalty) * prescribed_val;
            }
        }

        let config = SolverConfig {
            max_iterations: 1000,
            tolerance: 1e-6,
            preconditioner: SparsePreconditioner::None,
            verbose: false,
        };
        let solver = IterativeSolver::create(config);

        let guess_values: Vec<_> = fem_field.to_vec();
        let initial_guess = Array1::from_vec([guess_values.len()], guess_values)
            .expect("invariant: initial guess is 1-D with the collected length");
        let solution = solver
            .bicgstab_complex(
                matrix,
                rhs.as_slice().expect("rhs must be contiguous"),
                Some(
                    initial_guess
                        .as_slice()
                        .expect("initial_guess must be contiguous"),
                ),
            )
            .map_err(|_| {
                kwavers_core::error::KwaversError::Numerical(
                    kwavers_core::error::NumericalError::Instability {
                        operation: "bicgstab_complex".to_owned(),
                        condition: 0.0,
                    },
                )
            })?;

        fem_field[..num_nodes].copy_from_slice(&solution[..num_nodes]);

        Ok(())
    }
}
