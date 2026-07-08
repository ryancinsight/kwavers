//! BEM linear system assembly and solution reconstruction.

use super::super::types::BemBoundaryCondition;
use super::BemBoundaryManager;
use kwavers_core::error::KwaversResult;
use kwavers_math::linear_algebra::sparse::CompressedSparseRowMatrix;
use ndarray::Array1;
use kwavers_math::fft::Complex64;

#[derive(Clone, Copy)]
enum NodeBc {
    None,
    Dirichlet(Complex64),
    Neumann(Complex64),
    Robin(f64, Complex64),
    Radiation,
}

impl BemBoundaryManager {
    /// Assemble the linear system Ax = b for the BEM solver.
    ///
    /// Combines H and G matrices according to the applied boundary conditions,
    /// moving known values to the RHS and placing unknowns in A.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn assemble_bem_system(
        &self,
        h_matrix: &CompressedSparseRowMatrix<Complex64>,
        g_matrix: &CompressedSparseRowMatrix<Complex64>,
        wavenumber: f64,
    ) -> KwaversResult<(CompressedSparseRowMatrix<Complex64>, Array1<Complex64>)> {
        let n = h_matrix.rows;
        let bc_map = self.build_bc_map(n);

        let mut a_values = Vec::new();
        let mut a_col_indices = Vec::new();
        let mut a_row_pointers = vec![0; n + 1];
        let mut b = Array1::from_elem(n, Complex64::default());
        let mut a_nnz = 0;

        for i in 0..n {
            a_row_pointers[i] = a_values.len();
            let mut row_entries: Vec<(usize, Complex64)> = Vec::new();

            let (h_vals, h_cols) = h_matrix.get_row(i);
            for (&val, &col) in h_vals.iter().zip(h_cols.iter()) {
                match bc_map[col] {
                    NodeBc::Dirichlet(p_known) => {
                        b[i] -= val * p_known;
                    }
                    NodeBc::Neumann(_) | NodeBc::Robin(_, _) | NodeBc::Radiation | NodeBc::None => {
                        row_entries.push((col, val));
                    }
                }
            }

            let (g_vals, g_cols) = g_matrix.get_row(i);
            for (&val, &col) in g_vals.iter().zip(g_cols.iter()) {
                match bc_map[col] {
                    NodeBc::Dirichlet(_) => {
                        row_entries.push((col, -val));
                    }
                    NodeBc::Neumann(q_known) => {
                        b[i] += val * q_known;
                    }
                    NodeBc::Robin(alpha, g_known) => {
                        b[i] += val * g_known;
                        row_entries.push((col, val * alpha));
                    }
                    NodeBc::Radiation => {
                        let ik = Complex64::new(0.0, wavenumber);
                        row_entries.push((col, -val * ik));
                    }
                    NodeBc::None => {}
                }
            }

            row_entries.sort_by_key(|e| e.0);
            for (col, val) in row_entries {
                // Coalesce into the previous entry only when it is the same column
                // within the current row. `last().copied()` makes the empty case a
                // `None` mismatch (no unwrap), and the row-pointer bound prevents
                // merging across the row boundary.
                if a_col_indices.last().copied() == Some(col) && a_values.len() > a_row_pointers[i]
                {
                    let last_idx = a_values.len() - 1;
                    a_values[last_idx] += val;
                } else {
                    a_values.push(val);
                    a_col_indices.push(col);
                    a_nnz += 1;
                }
            }
        }
        a_row_pointers[n] = a_values.len();

        let a_matrix = CompressedSparseRowMatrix {
            rows: n,
            cols: n,
            values: a_values,
            col_indices: a_col_indices,
            row_pointers: a_row_pointers,
            nnz: a_nnz,
        };

        Ok((a_matrix, b))
    }

    /// Reconstruct full pressure and velocity fields from the BEM solution vector.
    #[must_use]
    pub fn reconstruct_solution(
        &self,
        x: &Array1<Complex64>,
        wavenumber: f64,
    ) -> (Array1<Complex64>, Array1<Complex64>) {
        let n = x.len();
        let bc_map = self.build_bc_map(n);
        let mut pressure = Array1::from_elem(n, Complex64::default());
        let mut velocity = Array1::from_elem(n, Complex64::default());

        for i in 0..n {
            match bc_map[i] {
                NodeBc::Dirichlet(p_known) => {
                    pressure[i] = p_known;
                    velocity[i] = x[i];
                }
                NodeBc::Neumann(q_known) => {
                    pressure[i] = x[i];
                    velocity[i] = q_known;
                }
                NodeBc::Robin(alpha, g_known) => {
                    pressure[i] = x[i];
                    velocity[i] = g_known - x[i] * alpha;
                }
                NodeBc::Radiation => {
                    pressure[i] = x[i];
                    velocity[i] = Complex64::new(0.0, wavenumber) * x[i];
                }
                NodeBc::None => {
                    pressure[i] = x[i];
                    velocity[i] = Complex64::default();
                }
            }
        }

        (pressure, velocity)
    }

    // ── Private helper ────────────────────────────────────────────────────────

    fn build_bc_map(&self, n: usize) -> Vec<NodeBc> {
        let mut bc_map = vec![NodeBc::None; n];
        for condition in &self.conditions {
            match condition {
                BemBoundaryCondition::Dirichlet(nodes) => {
                    for &(idx, val) in nodes {
                        if idx < n {
                            bc_map[idx] = NodeBc::Dirichlet(val);
                        }
                    }
                }
                BemBoundaryCondition::Neumann(nodes) => {
                    for &(idx, val) in nodes {
                        if idx < n {
                            bc_map[idx] = NodeBc::Neumann(val);
                        }
                    }
                }
                BemBoundaryCondition::Robin(nodes) => {
                    for &(idx, alpha, val) in nodes {
                        if idx < n {
                            bc_map[idx] = NodeBc::Robin(alpha, val);
                        }
                    }
                }
                BemBoundaryCondition::Radiation(nodes) => {
                    for &idx in nodes {
                        if idx < n {
                            bc_map[idx] = NodeBc::Radiation;
                        }
                    }
                }
            }
        }
        bc_map
    }
}
