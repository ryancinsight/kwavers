//! BEM system solve, FEM matrix assembly, and linear solver.

use kwavers_math::fft::Complex64;
use leto::Array1;

use kwavers_core::error::KwaversResult;
use kwavers_math::linear_algebra::sparse::solver::SparsePreconditioner;
use kwavers_math::linear_algebra::sparse::{
    CompressedSparseRowMatrix, CoordinateMatrix, IterativeSolver, SolverConfig,
};
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

    /// Assemble the FEM Helmholtz system matrix for the given `wavenumber`.
    ///
    /// Uses linear tetrahedral elements (P1). Each element contributes to the
    /// stiffness matrix `K` (grad-grad) and the mass matrix `M` (N·N), combined
    /// as `K − k² M`. Interface nodes receive a penalty row for Dirichlet
    /// enforcement.
    ///
    /// # Errors
    /// Returns `Err` when a tetrahedral Jacobian is singular.
    pub(crate) fn assemble_system_matrix(
        &self,
        fem_mesh: &TetrahedralMesh,
        wavenumber: f64,
    ) -> KwaversResult<CompressedSparseRowMatrix<Complex64>> {
        let num_nodes = fem_mesh.nodes.len();
        let mut coo = CoordinateMatrix::create(num_nodes, num_nodes);

        // Reference-element shape-function gradients for linear tetrahedra.
        let grad_ref = [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        for element in &fem_mesh.elements {
            let n_indices = element.nodes;
            let p0 = fem_mesh.nodes[n_indices[0]].coordinates;
            let p1 = fem_mesh.nodes[n_indices[1]].coordinates;
            let p2 = fem_mesh.nodes[n_indices[2]].coordinates;
            let p3 = fem_mesh.nodes[n_indices[3]].coordinates;

            let c0 = vec3_sub(p1, p0);
            let c1 = vec3_sub(p2, p0);
            let c2 = vec3_sub(p3, p0);
            let jacobian = mat3_from_columns(c0, c1, c2);

            if let Some(inv_j) = mat3_inv(&jacobian) {
                let inv_j_t = mat3_transpose(&inv_j);
                let det_j = mat3_det(&jacobian).abs();
                let volume = det_j / 6.0;

                let mut grads = [[0.0; 3]; 4];
                for k in 0..4 {
                    grads[k] = mat3_vec_mul(&inv_j_t, &grad_ref[k]);
                }

                for i in 0..4 {
                    for j in 0..4 {
                        let k_val = vec3_dot(grads[i], grads[j]) * volume;
                        let delta = if i == j { 1.0 } else { 0.0 };
                        let m_val = (1.0 + delta) * volume / 20.0;
                        let val =
                            Complex64::from(k_val) - Complex64::from(wavenumber.powi(2) * m_val);
                        coo.add_triplet(n_indices[i], n_indices[j], val);
                    }
                }
            } else {
                return Err(kwavers_core::error::KwaversError::Numerical(
                    kwavers_core::error::NumericalError::SingularMatrix {
                        operation: "element_jacobian".to_owned(),
                        condition_number: 0.0,
                    },
                ));
            }
        }

        // Penalty enforcement for Dirichlet interface nodes.
        let penalty = 1.0e14;
        for &node_idx in &self.interface.fem_interface_nodes {
            if node_idx < num_nodes {
                coo.add_triplet(node_idx, node_idx, Complex64::from(penalty));
            }
        }

        Ok(coo.to_csr())
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
        let solution = solver.bicgstab_complex(matrix, rhs.view(), Some(initial_guess.view()))?;

        for i in 0..num_nodes {
            fem_field[i] = solution[i];
        }

        Ok(())
    }
}

type Mat3 = [[f64; 3]; 3];

#[inline]
fn vec3_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn vec3_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

#[inline]
fn mat3_from_columns(c0: [f64; 3], c1: [f64; 3], c2: [f64; 3]) -> Mat3 {
    [
        [c0[0], c1[0], c2[0]],
        [c0[1], c1[1], c2[1]],
        [c0[2], c1[2], c2[2]],
    ]
}

#[inline]
fn mat3_det(m: &Mat3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

#[inline]
fn mat3_transpose(m: &Mat3) -> Mat3 {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

#[inline]
fn mat3_vec_mul(m: &Mat3, v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0].mul_add(v[0], m[0][1].mul_add(v[1], m[0][2] * v[2])),
        m[1][0].mul_add(v[0], m[1][1].mul_add(v[1], m[1][2] * v[2])),
        m[2][0].mul_add(v[0], m[2][1].mul_add(v[1], m[2][2] * v[2])),
    ]
}

fn mat3_inv(m: &Mat3) -> Option<Mat3> {
    let det = mat3_det(m);
    if det.abs() < 1e-14 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}
