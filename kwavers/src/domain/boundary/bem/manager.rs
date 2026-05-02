//! `BemBoundaryManager` — applies Dirichlet, Neumann, Robin, and radiation BCs
//! to the boundary integral equations.
//!
//! ## Mathematical Formulation
//!
//! ```text
//! BIE: c(p) + ∫_Γ G·∂p/∂n dΓ = ∫_Γ p·∂G/∂n dΓ
//! ```
//!
//! - **Dirichlet (Γ_D)**: p = g_D
//! - **Neumann (Γ_N)**: ∂p/∂n = g_N
//! - **Robin (Γ_R)**: ∂p/∂n + αp = g_R
//! - **Radiation (Γ_∞)**: Sommerfeld — ∂p/∂n − ikp ≈ 0

use super::types::BemBoundaryCondition;
use crate::core::error::KwaversResult;
use crate::math::linear_algebra::sparse::CompressedSparseRowMatrix;
use ndarray::Array1;
use num_complex::Complex64;

/// BEM boundary condition manager for boundary element solvers.
#[derive(Debug)]
pub struct BemBoundaryManager {
    conditions: Vec<BemBoundaryCondition>,
}

impl BemBoundaryManager {
    /// Create new boundary manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            conditions: Vec::new(),
        }
    }

    /// Clear all boundary conditions.
    pub fn clear(&mut self) {
        self.conditions.clear();
    }

    /// Add Dirichlet boundary condition.
    pub fn add_dirichlet(&mut self, node_values: Vec<(usize, Complex64)>) {
        self.conditions
            .push(BemBoundaryCondition::Dirichlet(node_values));
    }

    /// Add Neumann boundary condition.
    pub fn add_neumann(&mut self, node_derivatives: Vec<(usize, Complex64)>) {
        self.conditions
            .push(BemBoundaryCondition::Neumann(node_derivatives));
    }

    /// Add Robin boundary condition.
    pub fn add_robin(&mut self, node_conditions: Vec<(usize, f64, Complex64)>) {
        self.conditions
            .push(BemBoundaryCondition::Robin(node_conditions));
    }

    /// Add radiation boundary condition.
    pub fn add_radiation(&mut self, nodes: Vec<usize>) {
        self.conditions.push(BemBoundaryCondition::Radiation(nodes));
    }

    /// Apply all boundary conditions to the BEM system.
    ///
    /// # Arguments
    /// * `h_matrix` — H matrix from BEM (modified in-place)
    /// * `g_matrix` — G matrix from BEM (modified in-place)
    /// * `boundary_values` — Known boundary values vector (modified in-place)
    /// * `wavenumber` — Wavenumber k for radiation conditions
    pub fn apply_all(
        &self,
        h_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        g_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        boundary_values: &mut Array1<Complex64>,
        wavenumber: f64,
    ) -> KwaversResult<()> {
        for condition in &self.conditions {
            match condition {
                BemBoundaryCondition::Dirichlet(nv) => {
                    self.apply_dirichlet(h_matrix, g_matrix, boundary_values, nv)?;
                }
                BemBoundaryCondition::Neumann(nd) => {
                    self.apply_neumann(h_matrix, g_matrix, boundary_values, nd)?;
                }
                BemBoundaryCondition::Robin(nc) => {
                    self.apply_robin(h_matrix, g_matrix, boundary_values, nc)?;
                }
                BemBoundaryCondition::Radiation(nodes) => {
                    self.apply_radiation(h_matrix, g_matrix, wavenumber, nodes)?;
                }
            }
        }
        Ok(())
    }

    // ── Private applicators ───────────────────────────────────────────────────

    /// Apply Dirichlet BCs (p = g): zero the row in H/G and store g in RHS.
    fn apply_dirichlet(
        &self,
        h_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        g_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        boundary_values: &mut Array1<Complex64>,
        node_values: &[(usize, Complex64)],
    ) -> KwaversResult<()> {
        for &(node_idx, bc_value) in node_values {
            h_matrix.zero_row(node_idx);
            g_matrix.zero_row(node_idx);
            boundary_values[node_idx] = bc_value;
        }
        Ok(())
    }

    /// Apply Neumann BCs (∂p/∂n = g): store the derivative value in RHS.
    fn apply_neumann(
        &self,
        _h_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        _g_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        boundary_values: &mut Array1<Complex64>,
        node_derivatives: &[(usize, Complex64)],
    ) -> KwaversResult<()> {
        for &(node_idx, deriv_value) in node_derivatives {
            boundary_values[node_idx] = deriv_value;
        }
        Ok(())
    }

    /// Apply Robin BCs (∂p/∂n + αp = g): modify H diagonal, store g in RHS.
    fn apply_robin(
        &self,
        h_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        _g_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        boundary_values: &mut Array1<Complex64>,
        node_conditions: &[(usize, f64, Complex64)],
    ) -> KwaversResult<()> {
        for &(node_idx, alpha, g_value) in node_conditions {
            h_matrix.set_diagonal(node_idx, Complex64::new(alpha, 0.0));
            boundary_values[node_idx] = g_value;
        }
        Ok(())
    }

    /// Apply radiation BCs (Sommerfeld ∂p/∂n − ikp ≈ 0): add −ik to H diagonal.
    fn apply_radiation(
        &self,
        h_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        _g_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        wavenumber: f64,
        nodes: &[usize],
    ) -> KwaversResult<()> {
        let radiation_term = Complex64::new(0.0, -wavenumber);
        for &node_idx in nodes {
            h_matrix.set_diagonal(node_idx, radiation_term);
        }
        Ok(())
    }

    // ── Public assembly helpers ───────────────────────────────────────────────

    /// Assemble the linear system Ax = b for the BEM solver.
    ///
    /// Combines H and G matrices according to the applied boundary conditions,
    /// moving known values to the RHS and placing unknowns in A.
    pub fn assemble_bem_system(
        &self,
        h_matrix: &CompressedSparseRowMatrix<Complex64>,
        g_matrix: &CompressedSparseRowMatrix<Complex64>,
        wavenumber: f64,
    ) -> KwaversResult<(CompressedSparseRowMatrix<Complex64>, Array1<Complex64>)> {
        let n = h_matrix.rows;

        #[derive(Clone, Copy)]
        enum NodeBc {
            None,
            Dirichlet(Complex64),
            Neumann(Complex64),
            Robin(f64, Complex64),
            Radiation,
        }

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

        let mut a_values = Vec::new();
        let mut a_col_indices = Vec::new();
        let mut a_row_pointers = vec![0; n + 1];
        let mut b = Array1::zeros(n);
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
                if !a_col_indices.is_empty()
                    && *a_col_indices.last().unwrap() == col
                    && a_values.len() > a_row_pointers[i]
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
    pub fn reconstruct_solution(
        &self,
        x: &Array1<Complex64>,
        wavenumber: f64,
    ) -> (Array1<Complex64>, Array1<Complex64>) {
        let n = x.len();
        let mut pressure = Array1::zeros(n);
        let mut velocity = Array1::zeros(n);

        #[derive(Clone, Copy)]
        enum NodeBc {
            None,
            Dirichlet(Complex64),
            Neumann(Complex64),
            Robin(f64, Complex64),
            Radiation,
        }

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
                    velocity[i] = g_known - alpha * x[i];
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

    /// Number of boundary conditions registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.conditions.len()
    }

    /// True if no boundary conditions are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }
}

impl Default for BemBoundaryManager {
    fn default() -> Self {
        Self::new()
    }
}
