//! FEM Helmholtz Solver Implementation
//!
//! **STATUS: STUB / INCOMPLETE**
//!
//! This is a simplified demonstration FEM implementation.
//! Full mesh integration and proper assembly are not yet implemented.
//! TODO_AUDIT: P1 - Complete FEM Helmholtz Solver - Implement full finite element discretization with proper mesh integration, element assembly, and boundary conditions
//! DEPENDS ON: domain/mesh/tetrahedral.rs, domain/boundary/fem_boundary.rs, math/linear_algebra/sparse_solvers.rs
//! MISSING: Element matrix assembly for tetrahedral elements with basis functions
//! MISSING: Proper mesh integration with quadrature rules
//! MISSING: Boundary condition enforcement (Dirichlet/Neumann/Robin)
//! MISSING: Sparse matrix storage and efficient solvers
//! MISSING: Higher-order polynomial basis functions
//! MISSING: Radiation boundary conditions for unbounded domains
//! SEVERITY: HIGH (critical for complex geometry acoustic simulations)
//! THEOREM: Galerkin method: Find u ∈ V such that a(u,v) = f(v) ∀v ∈ V for variational form
//! THEOREM: Helmholtz weak form: ∫ (∇u·∇v - k²uv) dΩ = ∫ ∂u/∂n v dΓ for boundary value problems
//! REFERENCES: Wu (1995) Pre-asymptotic error analysis of FEM; Ihlenburg (1998) FEM for Helmholtz equation
//!
//! Core solver for the Helmholtz equation using finite element discretization.
//! Provides high-fidelity solutions for complex geometries where Born series
//! approximations fail.

use crate::core::error::{KwaversError, KwaversResult, NumericalError};
use super::assembly::FemAssembly;
use crate::domain::boundary::FemBoundaryManager;
use crate::domain::medium::Medium;
use crate::domain::mesh::TetrahedralMesh;
use crate::math::linear_algebra::sparse::csr::CompressedSparseRowMatrix;
use crate::math::linear_algebra::sparse::solver::{IterativeSolver, Preconditioner, SolverConfig};
use nalgebra::{Matrix3, Vector3};
use ndarray::{Array1, Array2, ArrayView2};
use num_complex::Complex64;

/// Finite Element Helmholtz solver configuration
#[derive(Debug, Clone)]
pub struct FemHelmholtzConfig {
    /// Polynomial degree for basis functions (1-4)
    pub polynomial_degree: usize,
    /// Wavenumber for Helmholtz equation
    pub wavenumber: f64,
    /// Solver tolerance
    pub tolerance: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Preconditioner type
    pub preconditioner: PreconditionerType,
    /// Use radiation boundary conditions
    pub radiation_boundary: bool,
}

/// Preconditioner options for FEM solver
#[derive(Debug, Clone, Copy)]
pub enum PreconditionerType {
    /// No preconditioning
    None,
    /// Diagonal preconditioning
    Diagonal,
    /// Incomplete LU factorization
    ILU,
    /// Algebraic multigrid
    AMG,
}

/// FEM Helmholtz solver for complex geometries
#[derive(Debug)]
pub struct FemHelmholtzSolver {
    /// Solver configuration
    config: FemHelmholtzConfig,
    /// Tetrahedral mesh
    mesh: TetrahedralMesh,
    /// Boundary condition manager
    boundary_manager: FemBoundaryManager,
    /// Global system matrix (sparse)
    system_matrix: CompressedSparseRowMatrix<Complex64>,
    /// Right-hand side vector
    #[allow(dead_code)]
    rhs: Array1<Complex64>,
    /// Solution vector
    solution: Array1<Complex64>,
}

impl FemHelmholtzSolver {
    /// Create new FEM Helmholtz solver
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

    /// Compute element stiffness and mass matrices for all elements
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
            // Get node coordinates
            let p0 = self.mesh.nodes[element.nodes[0]].coordinates;
            let p1 = self.mesh.nodes[element.nodes[1]].coordinates;
            let p2 = self.mesh.nodes[element.nodes[2]].coordinates;
            let p3 = self.mesh.nodes[element.nodes[3]].coordinates;

            let v0 = Vector3::from(p0);
            let v1 = Vector3::from(p1);
            let v2 = Vector3::from(p2);
            let v3 = Vector3::from(p3);

            // Jacobian matrix J = [x1-x0, x2-x0, x3-x0]
            // Note: This maps reference element (0,0,0)-(1,0,0)-(0,1,0)-(0,0,1) to physical tetrahedron
            let j_mat = Matrix3::from_columns(&[v1 - v0, v2 - v0, v3 - v0]);

            // Determinant of Jacobian
            let det_j = j_mat.determinant();

            // Volume of tetrahedron
            let volume = det_j.abs() / 6.0;

            if volume < 1e-14 {
                return Err(KwaversError::Numerical(NumericalError::SingularMatrix {
                    operation: "element_assembly".to_string(),
                    condition_number: 0.0,
                }));
            }

            // Inverse Jacobian transpose for gradient transformation
            // ∇_x = J^{-T} ∇_ξ
            let j_inv_t = j_mat
                .try_inverse()
                .ok_or(KwaversError::Numerical(NumericalError::SingularMatrix {
                    operation: "jacobian_inverse".to_string(),
                    condition_number: 0.0,
                }))?
                .transpose();

            // Gradients of basis functions in reference coordinates
            // φ1 = 1 - ξ - η - ζ, ∇φ1 = [-1, -1, -1]
            // φ2 = ξ,             ∇φ2 = [1, 0, 0]
            // φ3 = η,             ∇φ3 = [0, 1, 0]
            // φ4 = ζ,             ∇φ4 = [0, 0, 1]
            let grad_phi_ref = [
                Vector3::new(-1.0, -1.0, -1.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
            ];

            // Compute gradients in physical coordinates: ∇φ_i = J^{-T} * ∇ξ_i
            let mut grad_phi_phys = [Vector3::zeros(); 4];
            for k in 0..4 {
                grad_phi_phys[k] = j_inv_t * grad_phi_ref[k];
            }

            // Element Stiffness Matrix K_ij = ∫ ∇φ_i · ∇φ_j dΩ = (∇φ_i · ∇φ_j) * Volume
            let mut k_elem = Array2::<Complex64>::zeros((4, 4));
            for r in 0..4 {
                for c in 0..4 {
                    let dot_prod = grad_phi_phys[r].dot(&grad_phi_phys[c]);
                    k_elem[[r, c]] = Complex64::from(dot_prod * volume);
                }
            }

            // Element Mass Matrix M_ij = ∫ φ_i φ_j dΩ
            // Analytical formula for linear tetrahedron:
            // M_ii = V/10, M_ij = V/20 (i != j)
            let mut m_elem = Array2::<Complex64>::zeros((4, 4));
            let v_over_20 = Complex64::from(volume / 20.0);
            for r in 0..4 {
                for c in 0..4 {
                    if r == c {
                        m_elem[[r, c]] = v_over_20 * 2.0;
                    } else {
                        m_elem[[r, c]] = v_over_20;
                    }
                }
            }

            // Element RHS (Zero for now)
            let f_elem = Array1::<Complex64>::zeros(4);

            element_stiffness.push(k_elem);
            element_mass.push(m_elem);
            element_rhs.push(f_elem);
        }

        Ok((element_stiffness, element_mass, element_rhs))
    }

    /// Assemble global system matrices
    pub fn assemble_system<M: Medium>(&mut self, _medium: &M) -> KwaversResult<()> {
        let (stiffness_vec, mass_vec, rhs_vec) = self.compute_element_matrices()?;

        let assembler = FemAssembly::new();
        let (mut k_global, mut m_global, rhs_global) = assembler
            .assemble_global_matrices_parallel(
                &self.mesh.elements,
                &stiffness_vec,
                &mass_vec,
                &rhs_vec,
            )?;

        // Combine into system matrix A = K - k^2 M
        // We iterate over the non-zero values of K and M.
        // Since they are assembled from the same elements and nodes, they should have
        // identical sparsity patterns.
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

        // Apply boundary conditions
        self.boundary_manager.apply_all(
            &mut self.system_matrix,
            &mut m_global, // Not used after this, but API requires it
            &mut self.rhs,
            self.config.wavenumber,
        )?;

        Ok(())
    }

    /// Solve the assembled system
    pub fn solve_system(&mut self) -> KwaversResult<()> {
        let preconditioner = match self.config.preconditioner {
            PreconditionerType::None => Preconditioner::None,
            PreconditionerType::Diagonal => Preconditioner::Jacobi,
            _ => Preconditioner::None, // Other preconditioners not yet supported in iterative solver
        };

        let config = SolverConfig {
            max_iterations: self.config.max_iterations,
            tolerance: self.config.tolerance,
            preconditioner,
            verbose: false,
        };

        let solver = IterativeSolver::create(config);

        // Use current solution as initial guess if it contains non-zeros
        let x0 = if self.solution.iter().any(|c| c.norm() > 0.0) {
            Some(self.solution.view())
        } else {
            None
        };

        self.solution = solver.bicgstab_complex(&self.system_matrix, self.rhs.view(), x0)?;

        Ok(())
    }

    /// Interpolate solution at query points using element shape functions
    pub fn interpolate_solution(
        &self,
        query_points: ArrayView2<f64>,
    ) -> KwaversResult<Array1<Complex64>> {
        let num_points = query_points.nrows();
        let mut results = Array1::zeros(num_points);

        for (i, row) in query_points.outer_iter().enumerate() {
            let point = [row[0], row[1], row[2]];

            // Find containing elements
            // We use the first found element
            let elements = self.mesh.locate_point(point);

            if let Some(&elem_idx) = elements.first() {
                let element = &self.mesh.elements[elem_idx];
                let nodes = element.nodes;

                let p0 = self.mesh.nodes[nodes[0]].coordinates;
                let p1 = self.mesh.nodes[nodes[1]].coordinates;
                let p2 = self.mesh.nodes[nodes[2]].coordinates;
                let p3 = self.mesh.nodes[nodes[3]].coordinates;

                // Compute barycentric coordinates (u, v, w, t)
                let (u, v, w, t) = self.compute_shape_functions(point, p0, p1, p2, p3)?;

                // Nodal values
                let val0 = self.solution[nodes[0]];
                let val1 = self.solution[nodes[1]];
                let val2 = self.solution[nodes[2]];
                let val3 = self.solution[nodes[3]];

                // Linear interpolation: N0*v0 + N1*v1 + N2*v2 + N3*v3
                // For P1 tetrahedron: N0=t, N1=u, N2=v, N3=w
                results[i] = val0 * Complex64::from(t)
                    + val1 * Complex64::from(u)
                    + val2 * Complex64::from(v)
                    + val3 * Complex64::from(w);
            } else {
                // Point outside mesh - return 0.0
                results[i] = Complex64::new(0.0, 0.0);
            }
        }

        Ok(results)
    }

    /// Compute barycentric coordinates (u, v, w, t) for a point in a tetrahedron
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

        // Jacobian matrix J = [b-a, c-a, d-a]
        let m = Matrix3::from_columns(&[b - a, c - a, d - a]);

        // Invert Jacobian to map from physical to reference coordinates
        let inv = m.try_inverse().ok_or_else(|| {
            KwaversError::Numerical(NumericalError::SingularMatrix {
                operation: "element_interpolation".to_string(),
                condition_number: 0.0, // Condition number not readily available
            })
        })?;

        let uvw = inv * (p - a);
        let u = uvw[0];
        let v = uvw[1];
        let w = uvw[2];
        let t = 1.0 - u - v - w;

        Ok((u, v, w, t))
    }

    /// Get mutable reference to boundary condition manager
    #[must_use]
    pub fn boundary_manager(&mut self) -> &mut FemBoundaryManager {
        &mut self.boundary_manager
    }

    /// Get reference to boundary condition manager
    #[must_use]
    pub fn boundary_manager_ref(&self) -> &FemBoundaryManager {
        &self.boundary_manager
    }

    /// Get current solution vector
    #[must_use]
    pub fn solution(&self) -> &Array1<Complex64> {
        &self.solution
    }
}

impl Default for FemHelmholtzConfig {
    fn default() -> Self {
        Self {
            polynomial_degree: 1,
            wavenumber: 1.0,
            tolerance: 1e-8,
            max_iterations: 1000,
            preconditioner: PreconditionerType::Diagonal,
            radiation_boundary: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh::BoundaryType;
    use ndarray::arr2;
    use approx::assert_relative_eq;

    #[test]
    fn test_assembly_one_element() {
        // 1. Create mesh
        let mut mesh = TetrahedralMesh::new();
        let n0 = mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior);
        let n1 = mesh.add_node([1.0, 0.0, 0.0], BoundaryType::Interior);
        let n2 = mesh.add_node([0.0, 1.0, 0.0], BoundaryType::Interior);
        let n3 = mesh.add_node([0.0, 0.0, 1.0], BoundaryType::Interior);
        mesh.add_element([n0, n1, n2, n3], 0).expect("Failed to add element");

        // 2. Config with k=0 (Laplace) to simplify check
        let mut config = FemHelmholtzConfig::default();
        config.wavenumber = 0.0;
        config.radiation_boundary = false; // Disable radiation BC to keep matrix pure K

        let mut solver = FemHelmholtzSolver::new(config, mesh);

        // 3. Assemble
        // Pass dummy medium. Since we ignore medium in assemble_system, we can pass a dummy.
        // We use HomogeneousMedium for convenience.
        use crate::domain::grid::Grid;
        use crate::domain::medium::HomogeneousMedium;
        let grid = Grid::new(2, 2, 2, 1.0, 1.0, 1.0).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);

        solver.assemble_system(&medium).expect("Assembly failed");

        // 4. Verify System Matrix (Stiffness K)
        let mat = &solver.system_matrix;

        assert_eq!(mat.rows, 4);
        assert_eq!(mat.cols, 4);

        // Check diagonal K_00 = 0.5
        let k00 = mat.get_diagonal(0).re;
        assert_relative_eq!(k00, 0.5, epsilon = 1e-10);

        // Check K_11 = 1/6
        let k11 = mat.get_diagonal(1).re;
        assert_relative_eq!(k11, 1.0 / 6.0, epsilon = 1e-10);

        // Check row sum = 0
        let (vals, _) = mat.get_row(0);
        let sum_real: f64 = vals.iter().map(|c| c.re).sum();
        assert_relative_eq!(sum_real, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_interpolate_solution_basic() {
        // 1. Create a simple mesh with one tetrahedron
        let mut mesh = TetrahedralMesh::new();
        // Nodes: origin and unit vectors
        let n0 = mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior); // Origin
        let n1 = mesh.add_node([1.0, 0.0, 0.0], BoundaryType::Interior); // X
        let n2 = mesh.add_node([0.0, 1.0, 0.0], BoundaryType::Interior); // Y
        let n3 = mesh.add_node([0.0, 0.0, 1.0], BoundaryType::Interior); // Z

        // Add element
        mesh.add_element([n0, n1, n2, n3], 0).expect("Failed to add element");

        // 2. Initialize Solver
        let config = FemHelmholtzConfig::default();
        let mut solver = FemHelmholtzSolver::new(config, mesh);

        // 3. Set solution manually (linear field u = x + 2y + 3z)
        // At n0 (0,0,0): 0
        // At n1 (1,0,0): 1
        // At n2 (0,1,0): 2
        // At n3 (0,0,1): 3
        solver.solution[n0] = Complex64::new(0.0, 0.0);
        solver.solution[n1] = Complex64::new(1.0, 0.0);
        solver.solution[n2] = Complex64::new(2.0, 0.0);
        solver.solution[n3] = Complex64::new(3.0, 0.0);

        // 4. Query points
        // p1: Center of face (0.25, 0.25, 0.25) -> sum should be 0.25*(1+2+3) + 0.25*0 = 1.5
        // Wait, barycentric coords check:
        // p = (0.25, 0.25, 0.25)
        // u = x = 0.25
        // v = y = 0.25
        // w = z = 0.25
        // t = 1 - u - v - w = 0.25
        // Val = t*0 + u*1 + v*2 + w*3 = 0.25*6 = 1.5

        let query_points = arr2(&[
            [0.25, 0.25, 0.25],
            [1.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
        ]);

        let result = solver.interpolate_solution(query_points.view())
            .expect("Interpolation failed");

        assert_eq!(result.len(), 3);

        // Check p1
        assert_relative_eq!(result[0].re, 1.5, epsilon = 1e-10);
        assert_relative_eq!(result[0].im, 0.0, epsilon = 1e-10);

        // Check p2
        assert_relative_eq!(result[1].re, 1.0, epsilon = 1e-10);

        // Check p3
        assert_relative_eq!(result[2].re, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_system_one_element_dirichlet() {
        // 1. Create mesh with one element
        let mut mesh = TetrahedralMesh::new();
        // Nodes: origin and unit vectors
        let n0 = mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior); // Origin
        let n1 = mesh.add_node([1.0, 0.0, 0.0], BoundaryType::Interior); // X
        let n2 = mesh.add_node([0.0, 1.0, 0.0], BoundaryType::Interior); // Y
        let n3 = mesh.add_node([0.0, 0.0, 1.0], BoundaryType::Interior); // Z

        mesh.add_element([n0, n1, n2, n3], 0).expect("Failed to add element");

        // 2. Setup Solver (Laplace)
        let mut config = FemHelmholtzConfig::default();
        config.wavenumber = 0.0;
        config.radiation_boundary = false;
        config.tolerance = 1e-10;

        let mut solver = FemHelmholtzSolver::new(config, mesh);

        // 3. Add Boundary Conditions
        // Set u = x:
        // n0 (0,0,0) -> 0
        // n1 (1,0,0) -> 1
        // n2 (0,1,0) -> 0
        // n3 (0,0,1) -> Should be 0 (minimizes Dirichlet energy for u = x + c*z)

        solver.boundary_manager().add_dirichlet(vec![
            (n0, Complex64::new(0.0, 0.0)),
            (n1, Complex64::new(1.0, 0.0)),
            (n2, Complex64::new(0.0, 0.0)),
        ]);

        // 4. Assemble
        use crate::domain::grid::Grid;
        use crate::domain::medium::HomogeneousMedium;
        let grid = Grid::new(2, 2, 2, 1.0, 1.0, 1.0).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);

        solver.assemble_system(&medium).expect("Assembly failed");

        // 5. Solve
        solver.solve_system().expect("Solve failed");

        // 6. Check Solution at n3
        let u3 = solver.solution()[n3];
        assert_relative_eq!(u3.re, 0.0, epsilon = 1e-6);
        assert_relative_eq!(u3.im, 0.0, epsilon = 1e-6);

        // Check Solution at n1 (should be preserved)
        let u1 = solver.solution()[n1];
        assert_relative_eq!(u1.re, 1.0, epsilon = 1e-6);
    }
}
