//! `FemHelmholtzSolver` — P1 tetrahedral FEM Helmholtz solver.
//!
//! ## Mathematical Foundation
//!
//! Galerkin discretization of ∇²u + k²u = −f on a tetrahedral mesh with P1 basis
//! functions (Ihlenburg 1998, §2.1):
//! ```text
//! a(u,v) = ∫_Ω (∇u·∇v − k²uv) dΩ = ∫_Ω fv dΩ + ∫_Γ (∂u/∂n)v dΓ
//! ```
//!
//! **Element matrices** for tetrahedron {p₀,p₁,p₂,p₃}:
//! - J = [p₁−p₀ | p₂−p₀ | p₃−p₀], V = |det J|/6
//! - K_ij = V · (∇φᵢ · ∇φⱼ) — stiffness
//! - M_ij = V/(10+10·δᵢⱼ) — consistent mass (analytical P1 formula)
//! - System: A = K − k²M, solve via BiCGSTAB
//!
//! ## References
//! - Ihlenburg F (1998). *Finite Element Analysis of Acoustic Scattering*. Springer.

use super::config::{FemHelmholtzConfig, PreconditionerType};
use crate::core::error::{KwaversError, KwaversResult, NumericalError};
use crate::domain::boundary::FemBoundaryManager;
use crate::domain::medium::Medium;
use crate::domain::mesh::TetrahedralMesh;
use crate::math::linear_algebra::sparse::csr::CompressedSparseRowMatrix;
use crate::math::linear_algebra::sparse::solver::{IterativeSolver, Preconditioner, SolverConfig};
use crate::solver::forward::helmholtz::fem::assembly::FemAssembly;
use nalgebra::{Matrix3, Vector3};
use ndarray::{Array1, Array2, ArrayView2};
use num_complex::Complex64;

/// Finite Element Helmholtz solver for complex geometries.
#[derive(Debug)]
pub struct FemHelmholtzSolver {
    config: FemHelmholtzConfig,
    mesh: TetrahedralMesh,
    boundary_manager: FemBoundaryManager,
    pub(super) system_matrix: CompressedSparseRowMatrix<Complex64>,
    #[allow(dead_code)]
    rhs: Array1<Complex64>,
    /// Nodal solution vector u_h ∈ ℂ^{n_dof}.
    pub solution: Array1<Complex64>,
}

impl FemHelmholtzSolver {
    /// Construct solver from configuration and mesh.
    pub fn new(config: FemHelmholtzConfig, mesh: TetrahedralMesh) -> Self {
        let num_dofs = mesh.nodes.len();
        Self {
            config,
            mesh,
            boundary_manager: FemBoundaryManager::new(),
            system_matrix: CompressedSparseRowMatrix::create(num_dofs, num_dofs),
            rhs: Array1::zeros(num_dofs),
            solution: Array1::zeros(num_dofs),
        }
    }

    /// Compute per-element stiffness K_e, consistent mass M_e, and RHS f_e.
    ///
    /// P1 reference-element gradients:
    /// ```text
    /// ∇ξ₀ = (−1,−1,−1), ∇ξ₁ = (1,0,0), ∇ξ₂ = (0,1,0), ∇ξ₃ = (0,0,1)
    /// ∇φᵢ = J^{−T} ∇ξᵢ
    /// ```
    fn compute_element_matrices(
        &self,
    ) -> KwaversResult<(
        Vec<Array2<Complex64>>,
        Vec<Array2<Complex64>>,
        Vec<Array1<Complex64>>,
    )> {
        let mut element_stiffness = Vec::with_capacity(self.mesh.elements.len());
        let mut element_mass = Vec::with_capacity(self.mesh.elements.len());
        let mut element_rhs = Vec::with_capacity(self.mesh.elements.len());

        for element in &self.mesh.elements {
            let p0 = self.mesh.nodes[element.nodes[0]].coordinates;
            let p1 = self.mesh.nodes[element.nodes[1]].coordinates;
            let p2 = self.mesh.nodes[element.nodes[2]].coordinates;
            let p3 = self.mesh.nodes[element.nodes[3]].coordinates;

            let v0 = Vector3::from(p0);
            let v1 = Vector3::from(p1);
            let v2 = Vector3::from(p2);
            let v3 = Vector3::from(p3);

            // Jacobian J = [p₁−p₀ | p₂−p₀ | p₃−p₀]
            let j_mat = Matrix3::from_columns(&[v1 - v0, v2 - v0, v3 - v0]);
            let det_j = j_mat.determinant();
            let volume = det_j.abs() / 6.0;

            if volume < 1e-14 {
                return Err(KwaversError::Numerical(NumericalError::SingularMatrix {
                    operation: "element_assembly".to_string(),
                    condition_number: 0.0,
                }));
            }

            let j_inv_t = j_mat
                .try_inverse()
                .ok_or(KwaversError::Numerical(NumericalError::SingularMatrix {
                    operation: "jacobian_inverse".to_string(),
                    condition_number: 0.0,
                }))?
                .transpose();

            let grad_phi_ref = [
                Vector3::new(-1.0, -1.0, -1.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
            ];

            let mut grad_phi_phys = [Vector3::zeros(); 4];
            for k in 0..4 {
                grad_phi_phys[k] = j_inv_t * grad_phi_ref[k];
            }

            // Stiffness: K_ij = V (∇φᵢ · ∇φⱼ)
            let mut k_elem = Array2::<Complex64>::zeros((4, 4));
            for r in 0..4 {
                for c in 0..4 {
                    k_elem[[r, c]] =
                        Complex64::from(grad_phi_phys[r].dot(&grad_phi_phys[c]) * volume);
                }
            }

            // Consistent mass: M_ii = V/10, M_ij = V/20 (i≠j)
            let mut m_elem = Array2::<Complex64>::zeros((4, 4));
            let v_over_20 = Complex64::from(volume / 20.0);
            for r in 0..4 {
                for c in 0..4 {
                    m_elem[[r, c]] = if r == c { v_over_20 * 2.0 } else { v_over_20 };
                }
            }

            element_stiffness.push(k_elem);
            element_mass.push(m_elem);
            element_rhs.push(Array1::<Complex64>::zeros(4));
        }

        Ok((element_stiffness, element_mass, element_rhs))
    }

    /// Assemble the global system matrix A = K − k²M and apply boundary conditions.
    ///
    /// The `medium` parameter is reserved for future heterogeneous-k support;
    /// currently the uniform `config.wavenumber` is used.
    pub fn assemble_system<M: Medium>(&mut self, _medium: &M) -> KwaversResult<()> {
        if self.mesh.elements.is_empty() {
            let num_nodes = self.mesh.nodes.len();
            let mut k_global = CompressedSparseRowMatrix::create(num_nodes, num_nodes);
            let mut m_global = CompressedSparseRowMatrix::create(num_nodes, num_nodes);
            let mut rhs_global = Array1::<Complex64>::zeros(num_nodes);

            self.boundary_manager.apply_all(
                &mut k_global,
                &mut m_global,
                &mut rhs_global,
                self.config.wavenumber,
            )?;

            self.system_matrix = k_global;
            self.rhs = rhs_global;
            return Ok(());
        }

        let (stiffness_vec, mass_vec, rhs_vec) = self.compute_element_matrices()?;

        let assembler = FemAssembly::new();
        let (mut k_global, mut m_global, rhs_global) = assembler
            .assemble_global_matrices_parallel(
                &self.mesh.elements,
                &stiffness_vec,
                &mass_vec,
                &rhs_vec,
            )?;

        let k_sq = Complex64::from(self.config.wavenumber.powi(2));

        if k_global.values.len() != m_global.values.len() {
            return Err(KwaversError::Numerical(NumericalError::Instability {
                operation: "matrix_combination".to_string(),
                condition: 0.0,
            }));
        }

        for (val_k, val_m) in k_global.values.iter_mut().zip(m_global.values.iter()) {
            *val_k -= k_sq * val_m;
        }

        self.system_matrix = k_global;
        self.rhs = rhs_global;

        self.boundary_manager.apply_all(
            &mut self.system_matrix,
            &mut m_global,
            &mut self.rhs,
            self.config.wavenumber,
        )?;

        Ok(())
    }

    /// Solve the assembled system via BiCGSTAB with the configured preconditioner.
    pub fn solve_system(&mut self) -> KwaversResult<()> {
        let preconditioner = match self.config.preconditioner {
            PreconditionerType::None => Preconditioner::None,
            PreconditionerType::Diagonal => Preconditioner::Jacobi,
            _ => Preconditioner::None,
        };

        let config = SolverConfig {
            max_iterations: self.config.max_iterations,
            tolerance: self.config.tolerance,
            preconditioner,
            verbose: false,
        };

        let solver = IterativeSolver::create(config);
        let x0 = if self.solution.iter().any(|c| c.norm() > 0.0) {
            Some(self.solution.view())
        } else {
            None
        };

        self.solution = solver.bicgstab_complex(&self.system_matrix, self.rhs.view(), x0)?;
        Ok(())
    }

    /// Interpolate the nodal solution at arbitrary query points via barycentric coordinates.
    ///
    /// Returns zero for query points outside the mesh domain.
    pub fn interpolate_solution(
        &self,
        query_points: ArrayView2<f64>,
    ) -> KwaversResult<Array1<Complex64>> {
        let num_points = query_points.nrows();
        let mut results = Array1::zeros(num_points);

        for (i, row) in query_points.outer_iter().enumerate() {
            let point = [row[0], row[1], row[2]];
            let elements = self.mesh.locate_point(point);

            if let Some(&elem_idx) = elements.first() {
                let element = &self.mesh.elements[elem_idx];
                let nodes = element.nodes;

                let p0 = self.mesh.nodes[nodes[0]].coordinates;
                let p1 = self.mesh.nodes[nodes[1]].coordinates;
                let p2 = self.mesh.nodes[nodes[2]].coordinates;
                let p3 = self.mesh.nodes[nodes[3]].coordinates;

                let (u, v, w, t) = self.compute_shape_functions(point, p0, p1, p2, p3)?;

                results[i] = self.solution[nodes[0]] * Complex64::from(t)
                    + self.solution[nodes[1]] * Complex64::from(u)
                    + self.solution[nodes[2]] * Complex64::from(v)
                    + self.solution[nodes[3]] * Complex64::from(w);
            }
            // Point outside mesh → result[i] stays 0.0
        }

        Ok(results)
    }

    /// Compute barycentric coordinates (u, v, w, t) for `point` inside tetrahedron {p0..p3}.
    ///
    /// Maps physical coordinates to reference coordinates via J^{-1}:
    /// ```text
    /// [u, v, w]ᵀ = J^{-1} (point − p₀),  t = 1 − u − v − w
    /// ```
    fn compute_shape_functions(
        &self,
        point: [f64; 3],
        p0: [f64; 3],
        p1: [f64; 3],
        p2: [f64; 3],
        p3: [f64; 3],
    ) -> KwaversResult<(f64, f64, f64, f64)> {
        let a = Vector3::new(p0[0], p0[1], p0[2]);
        let b = Vector3::new(p1[0], p1[1], p1[2]);
        let c = Vector3::new(p2[0], p2[1], p2[2]);
        let d = Vector3::new(p3[0], p3[1], p3[2]);
        let p = Vector3::new(point[0], point[1], point[2]);

        let m = Matrix3::from_columns(&[b - a, c - a, d - a]);
        let inv = m.try_inverse().ok_or_else(|| {
            KwaversError::Numerical(NumericalError::SingularMatrix {
                operation: "element_interpolation".to_string(),
                condition_number: 0.0,
            })
        })?;

        let uvw = inv * (p - a);
        let u = uvw[0];
        let v = uvw[1];
        let w = uvw[2];
        let t = 1.0 - u - v - w;

        Ok((u, v, w, t))
    }

    /// Mutable reference to the boundary condition manager.
    #[must_use]
    pub fn boundary_manager(&mut self) -> &mut FemBoundaryManager {
        &mut self.boundary_manager
    }

    /// Immutable reference to the boundary condition manager.
    #[must_use]
    pub fn boundary_manager_ref(&self) -> &FemBoundaryManager {
        &self.boundary_manager
    }

    /// Nodal solution vector u_h.
    #[must_use]
    pub fn solution(&self) -> &Array1<Complex64> {
        &self.solution
    }
}
