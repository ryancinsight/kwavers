use super::BemSolver;
use crate::forward::bem::{
    field::BemSolution,
    geometry::{point_to_triangle_distance, triangle_characteristic_length},
    integrals::{compute_nearfield_integrals, compute_nonsingular_integrals},
};
use kwavers_boundary::BemBoundaryManager;
use kwavers_core::error::KwaversResult;
use kwavers_math::linear_algebra::sparse::{
    solver::SparsePreconditioner, CompressedSparseRowMatrix,
};
use moirai_parallel::{map_collect_with, Adaptive};
use ndarray::Array1;
use kwavers_math::fft::Complex64;

impl BemSolver {
    /// Invalidate cached system matrices (called when wavenumber changes).
    pub fn invalidate_matrix(&mut self) {
        self.h_matrix = None;
        self.g_matrix = None;
    }

    /// Get mutable reference to boundary condition manager
    #[must_use]
    pub fn boundary_manager(&mut self) -> &mut BemBoundaryManager {
        &mut self.boundary_manager
    }

    /// Get reference to boundary condition manager
    #[must_use]
    pub fn boundary_manager_ref(&self) -> &BemBoundaryManager {
        &self.boundary_manager
    }

    /// Get local BEM node index for a global mesh node index
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn local_index(&self, global_idx: usize) -> Option<usize> {
        self.global_to_local_node.get(&global_idx).copied()
    }

    /// Solve the rigid-scattering CFIE for a prescribed incident field.
    ///
    /// Solves the Burton–Miller CFIE:
    ///   (H + α·D)·p = (G + α·(0.5I + H'))·∂p/∂n_inc
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn solve_rigid(
        &mut self,
        p_inc: Vec<Complex64>,
        dp_inc_dn: Vec<Complex64>,
    ) -> KwaversResult<Vec<Complex64>> {
        let n = self.vertices.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        if self.h_matrix.is_none() || self.g_matrix.is_none() {
            self.assemble_system()?;
        }

        let h_mat = self.h_matrix.as_ref().unwrap();
        let g_mat = self.g_matrix.as_ref().unwrap();
        let alpha = self.config.coupling_alpha;

        let mut rhs = vec![Complex64::new(0.0, 0.0); n];
        for (rhs_elem, window) in rhs.iter_mut().zip(g_mat.row_pointers.windows(2)) {
            let (row_start, row_end) = (window[0], window[1]);
            for ptr in row_start..row_end {
                let j = g_mat.col_indices[ptr];
                if j < dp_inc_dn.len() {
                    *rhs_elem += g_mat.values[ptr] * dp_inc_dn[j];
                }
            }
        }

        let mut a_values = h_mat.values.clone();
        for i in 0..n {
            let diag_ptr = h_mat.row_pointers[i];
            let row_end = h_mat.row_pointers[i + 1];
            for (a_val, &col_idx) in a_values[diag_ptr..row_end]
                .iter_mut()
                .zip(&h_mat.col_indices[diag_ptr..row_end])
            {
                if col_idx == i {
                    *a_val += alpha * Complex64::new(0.5, 0.0);
                    break;
                }
            }
        }

        let a_matrix = CompressedSparseRowMatrix {
            rows: h_mat.rows,
            cols: h_mat.cols,
            values: a_values,
            col_indices: h_mat.col_indices.clone(),
            row_pointers: h_mat.row_pointers.clone(),
            nnz: h_mat.nnz,
        };

        let rhs_arr = Array1::from_vec(rhs);
        let solver_config = kwavers_math::linear_algebra::sparse::solver::SolverConfig {
            max_iterations: self.config.max_iterations,
            tolerance: self.config.tolerance,
            preconditioner: SparsePreconditioner::None,
            verbose: false,
        };
        let solver = kwavers_math::linear_algebra::sparse::IterativeSolver::create(solver_config);
        let p_scat = solver.bicgstab_complex(&a_matrix, rhs_arr.view(), None)?;

        let p_total: Vec<Complex64> = p_inc
            .iter()
            .zip(p_scat.iter())
            .map(|(&pi, &ps)| pi + ps)
            .collect();

        Ok(p_total)
    }

    /// Solve the BEM system applying boundary conditions.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn solve(
        &mut self,
        wavenumber: f64,
        source_terms: Option<&Array1<Complex64>>,
    ) -> KwaversResult<BemSolution> {
        if self.h_matrix.is_none() || self.g_matrix.is_none() {
            self.config.wavenumber = wavenumber;
            self.assemble_system()?;
        }

        let h_matrix = self.h_matrix.as_ref().unwrap();
        let g_matrix = self.g_matrix.as_ref().unwrap();

        let (a_matrix, mut b_vector) = self
            .boundary_manager
            .assemble_bem_system(h_matrix, g_matrix, wavenumber)?;

        if let Some(sources) = source_terms {
            b_vector += sources;
        }

        let x = self.solve_bem_system(&a_matrix, &b_vector)?;

        let (boundary_pressure, boundary_velocity) =
            self.boundary_manager.reconstruct_solution(&x, wavenumber);

        Ok(BemSolution {
            boundary_pressure,
            boundary_velocity,
            wavenumber,
        })
    }

    /// Compute scattered field at evaluation points using the BEM representation formula.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn compute_scattered_field(
        &self,
        evaluation_points: &Array1<[f64; 3]>,
        solution: &BemSolution,
    ) -> KwaversResult<Array1<Complex64>> {
        let k = solution.wavenumber;

        let points_slice = evaluation_points.as_slice().ok_or_else(|| {
            kwavers_core::error::KwaversError::InvalidInput(
                "Evaluation points array must be contiguous for parallel processing".to_owned(),
            )
        })?;

        let results: Vec<Complex64> =
            map_collect_with::<Adaptive, _, _, _>(points_slice, |&r_eval| {
                let mut total_field = Complex64::new(0.0, 0.0);

                for element_indices in &self.triangles {
                    let n1 = element_indices[0];
                    let n2 = element_indices[1];
                    let n3 = element_indices[2];

                    let p1 = self.vertices[n1];
                    let p2 = self.vertices[n2];
                    let p3 = self.vertices[n3];

                    let distance = point_to_triangle_distance(r_eval, p1, p2, p3);
                    let element_size = triangle_characteristic_length(p1, p2, p3);
                    let (h_res, g_res) = if distance < 0.2 * element_size {
                        compute_nearfield_integrals(k, r_eval, [p1, p2, p3], distance, element_size)
                    } else {
                        compute_nonsingular_integrals(k, r_eval, [p1, p2, p3])
                    };

                    let u_vals = [
                        solution.boundary_pressure[n1],
                        solution.boundary_pressure[n2],
                        solution.boundary_pressure[n3],
                    ];

                    let q_vals = [
                        solution.boundary_velocity[n1],
                        solution.boundary_velocity[n2],
                        solution.boundary_velocity[n3],
                    ];

                    for m in 0..3 {
                        total_field += g_res[m] * q_vals[m] - h_res[m] * u_vals[m];
                    }
                }

                total_field
            });

        Ok(Array1::from_vec(results))
    }
}
