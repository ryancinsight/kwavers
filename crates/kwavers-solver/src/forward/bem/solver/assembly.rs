use super::BemSolver;
use crate::forward::bem::integrals::{compute_nonsingular_integrals, compute_singular_integrals};
use kwavers_core::error::KwaversResult;
use kwavers_math::fft::Complex64;
use kwavers_math::linear_algebra::sparse::{
    solver::{IterativeSolver, SolverConfig, SparsePreconditioner},
    CompressedSparseRowMatrix,
};
use leto::Array1;

impl BemSolver {
    /// Assemble BEM system matrices
    ///
    /// Computes the boundary integrals to assemble the H and G matrices.
    /// Uses standard Gaussian quadrature for non-singular elements and
    /// Duffy transformation / singularity handling for singular elements.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn assemble_system(&mut self) -> KwaversResult<()> {
        let n = self.vertices.len();
        if n == 0 {
            return Ok(());
        }

        let mut h_values = vec![Complex64::new(0.0, 0.0); n * n];
        let mut g_values = vec![Complex64::new(0.0, 0.0); n * n];
        let idx = |row: usize, col: usize| row * n + col;
        let k = self.config.wavenumber;

        for i in 0..n {
            let r_i = self.vertices[i];

            for element in &self.triangles {
                let node_indices = element;
                let p1 = self.vertices[node_indices[0]];
                let p2 = self.vertices[node_indices[1]];
                let p3 = self.vertices[node_indices[2]];

                let singular_idx = node_indices.iter().position(|&idx| idx == i);

                let (h_contrib, g_contrib) = if let Some(vertex_idx) = singular_idx {
                    compute_singular_integrals(k, r_i, [p1, p2, p3], vertex_idx)
                } else {
                    compute_nonsingular_integrals(k, r_i, [p1, p2, p3])
                };

                for m in 0..3 {
                    let col = node_indices[m];
                    h_values[idx(i, col)] += h_contrib[m];
                    g_values[idx(i, col)] += g_contrib[m];
                }
            }

            h_values[idx(i, i)] += Complex64::new(0.5, 0.0);
        }

        let mut col_indices = Vec::with_capacity(n * n);
        for _ in 0..n {
            for c in 0..n {
                col_indices.push(c);
            }
        }

        let mut row_pointers = Vec::with_capacity(n + 1);
        for i in 0..=n {
            row_pointers.push(i * n);
        }

        self.h_matrix = Some(CompressedSparseRowMatrix {
            rows: n,
            cols: n,
            values: h_values,
            col_indices: col_indices.clone(),
            row_pointers: row_pointers.clone(),
            nnz: n * n,
        });

        self.g_matrix = Some(CompressedSparseRowMatrix {
            rows: n,
            cols: n,
            values: g_values,
            col_indices,
            row_pointers,
            nnz: n * n,
        });

        Ok(())
    }

    /// Solve the assembled BEM linear system via BiCGSTAB
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn solve_bem_system(
        &self,
        a_matrix: &CompressedSparseRowMatrix<Complex64>,
        b_vector: &Array1<Complex64>,
    ) -> KwaversResult<Array1<Complex64>> {
        let solver_config = SolverConfig {
            max_iterations: self.config.max_iterations,
            tolerance: self.config.tolerance,
            preconditioner: SparsePreconditioner::None,
            verbose: false,
        };
        let solver = IterativeSolver::create(solver_config);
        solver.bicgstab_complex(a_matrix, b_vector.view(), None)
    }
}
