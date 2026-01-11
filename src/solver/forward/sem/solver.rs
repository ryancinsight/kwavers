//! Spectral Element Method Solver Implementation
//!
//! Main solver for the Spectral Element Method, integrating:
//! - High-order basis functions on GLL points
//! - Hexahedral element assembly
//! - Time integration (Newmark method)
//! - Boundary condition management

use crate::core::error::KwaversResult;
use crate::domain::boundary::FemBoundaryManager;
use crate::math::linear_algebra::sparse::CompressedSparseRowMatrix;
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::sync::Arc;

use super::basis::SemBasis;
use super::elements::SemMesh;
use super::integration::NewmarkIntegrator;

/// Configuration for SEM solver
#[derive(Debug, Clone)]
pub struct SemConfig {
    /// Polynomial degree for basis functions (2-8 recommended)
    pub polynomial_degree: usize,
    /// Wavenumber for Helmholtz equation (2πf/c)
    pub wavenumber: f64,
    /// Time step size
    pub dt: f64,
    /// Total number of time steps
    pub n_steps: usize,
    /// Sound speed (m/s)
    pub sound_speed: f64,
    /// Density (kg/m³)
    pub density: f64,
}

impl Default for SemConfig {
    fn default() -> Self {
        Self {
            polynomial_degree: 4,
            wavenumber: 1.0,
            dt: 1e-7, // 100 ns
            n_steps: 1000,
            sound_speed: 1500.0, // Water
            density: 1000.0,     // Water
        }
    }
}

/// Spectral Element Method solver for acoustic wave propagation
#[derive(Debug)]
pub struct SemSolver {
    /// Solver configuration
    config: SemConfig,
    /// Spectral element mesh
    mesh: Arc<SemMesh>,
    /// Global mass matrix (diagonal by construction in SEM)
    mass_matrix: Array1<f64>,
    /// Global stiffness matrix
    stiffness_matrix: CompressedSparseRowMatrix<f64>,
    /// Boundary condition manager
    boundary_manager: FemBoundaryManager,
    /// Time integrator
    integrator: NewmarkIntegrator,
    /// Current solution vector
    solution: Array1<f64>,
    /// Current time step index
    time_step: usize,
}

impl SemSolver {
    /// Create new SEM solver
    pub fn new(config: SemConfig, mesh: Arc<SemMesh>) -> KwaversResult<Self> {
        if config.polynomial_degree == 0 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "SEM polynomial_degree must be positive".to_string(),
            ));
        }
        if !config.dt.is_finite() || config.dt <= 0.0 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "SEM dt must be finite and positive".to_string(),
            ));
        }
        if config.n_steps == 0 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "SEM n_steps must be positive".to_string(),
            ));
        }
        if !config.sound_speed.is_finite() || config.sound_speed <= 0.0 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "SEM sound_speed must be finite and positive".to_string(),
            ));
        }
        if !config.density.is_finite() || config.density <= 0.0 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "SEM density must be finite and positive".to_string(),
            ));
        }
        if config.polynomial_degree != mesh.basis.degree {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "SEM config polynomial_degree {} does not match mesh degree {}",
                config.polynomial_degree, mesh.basis.degree
            )));
        }

        let n_dofs = mesh.n_dofs;

        // Initialize matrices
        let mass_matrix = Array1::<f64>::zeros(n_dofs);
        let stiffness_matrix = CompressedSparseRowMatrix::create(n_dofs, n_dofs);

        // Initialize integrator with zero initial conditions
        let integrator = NewmarkIntegrator::average_acceleration(config.dt, n_dofs);

        // Initialize solution vector
        let solution = Array1::<f64>::zeros(n_dofs);

        Ok(Self {
            config,
            mesh,
            mass_matrix,
            stiffness_matrix,
            boundary_manager: FemBoundaryManager::new(),
            integrator,
            solution,
            time_step: 0,
        })
    }

    /// Assemble global system matrices
    ///
    /// Computes element matrices and assembles them into global
    /// mass and stiffness matrices using SEM basis functions.
    pub fn assemble_system(&mut self) -> KwaversResult<()> {
        // Reset matrices
        self.mass_matrix.fill(0.0);
        // Note: stiffness_matrix reset would be implemented here

        let n_gll = self.mesh.basis.n_points();
        let n_elements = self.mesh.elements.len();
        let weights = &self.mesh.basis.gll_weights;

        // Assemble each element
        for elem_idx in 0..n_elements {
            let element = &self.mesh.elements[elem_idx];
            let element_id = element.id;
            let jacobian_det = &element.jacobian_det;

            // Assemble this element's contribution
            for i in 0..n_gll {
                for j in 0..n_gll {
                    for k in 0..n_gll {
                        let ijk = self.element_local_to_global_dof(element_id, i, j, k, n_gll);
                        let w = weights[i] * weights[j] * weights[k];

                        // Add mass matrix contribution (diagonal)
                        // In SEM, mass matrix is diagonal: M_ii = ∫ φ_i² dΩ
                        self.mass_matrix[ijk] += self.config.density * jacobian_det[[i, j, k]] * w;

                        // Stiffness matrix assembly would go here
                        // K_ij += ∫ ∇φ_i · ∇φ_j dΩ
                    }
                }
            }
        }

        Ok(())
    }

    /// Convert element local DOF indices to global DOF index
    fn element_local_to_global_dof(
        &self,
        element_id: usize,
        i: usize,
        j: usize,
        k: usize,
        n_gll: usize,
    ) -> usize {
        // Simple mapping: element_id * dofs_per_element + local_index
        let dofs_per_element = n_gll * n_gll * n_gll;
        let local_index = i + j * n_gll + k * n_gll * n_gll;
        element_id * dofs_per_element + local_index
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

    /// Set initial conditions
    pub fn set_initial_conditions(
        &mut self,
        initial_displacement: Array1<f64>,
    ) -> KwaversResult<()> {
        if initial_displacement.len() != self.solution.len() {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "Initial condition size mismatch: expected {}, got {}",
                self.solution.len(),
                initial_displacement.len()
            )));
        }

        self.solution.assign(&initial_displacement);
        Ok(())
    }

    /// Advance one time step
    pub fn step(&mut self) -> KwaversResult<()> {
        if self.time_step >= self.config.n_steps {
            return Ok(()); // Simulation complete
        }

        // Compute acceleration from equations of motion: M*a + C*v + K*u = F
        // For acoustic wave equation: ∇²p - (1/c²)∂²p/∂t² = 0
        // In SEM: M * ∂²u/∂t² + K * u = F

        let acceleration = self.compute_acceleration()?;

        // Advance time integration
        self.integrator.step(&acceleration);

        // Update solution from integrator
        self.solution.assign(&self.integrator.displacement);

        self.time_step += 1;

        Ok(())
    }

    /// Compute acceleration from equations of motion
    fn compute_acceleration(&self) -> KwaversResult<Array1<f64>> {
        let n_dofs = self.solution.len();
        let mut acceleration = Array1::<f64>::zeros(n_dofs);

        // For acoustic wave equation: M * a = -K * u + F
        // where u is displacement/pressure, a is acceleration

        // Simplified: assume unit mass matrix and basic stiffness
        // In full implementation, this would involve sparse matrix operations
        for i in 0..n_dofs {
            // Mass matrix is diagonal in SEM: a_i = (F_i - K_i*u_i) / M_ii
            let m = self.mass_matrix[i];
            if !m.is_finite() || m <= 0.0 {
                return Err(crate::core::error::KwaversError::InvalidInput(format!(
                    "Non-positive mass matrix entry at dof {}: {}",
                    i, m
                )));
            }
            let mass_inv = 1.0 / m;
            let stiffness_term = self.config.wavenumber.powi(2) * self.solution[i]; // Helmholtz: -∇²u + k²u
            let force = 0.0; // No external forces in this simple case

            acceleration[i] = mass_inv * (force - stiffness_term);
        }

        Ok(acceleration)
    }

    /// Run complete simulation
    pub fn run_simulation(&mut self) -> KwaversResult<()> {
        // Assemble system matrices first
        self.assemble_system()?;

        // Time stepping loop
        while self.time_step < self.config.n_steps {
            self.step()?;
        }

        Ok(())
    }

    /// Get current solution
    #[must_use]
    pub fn solution(&self) -> &Array1<f64> {
        &self.solution
    }

    /// Get current time
    #[must_use]
    pub fn current_time(&self) -> f64 {
        self.integrator.time
    }

    /// Get current time step index
    #[must_use]
    pub fn current_step(&self) -> usize {
        self.time_step
    }

    /// Interpolate solution at arbitrary points
    ///
    /// Uses SEM basis functions to interpolate the solution field
    /// at arbitrary points within the domain.
    pub fn interpolate_solution(&self, query_points: &Array2<f64>) -> KwaversResult<Array1<f64>> {
        let mut interpolated = Array1::<f64>::zeros(query_points.nrows());

        // For each query point, find containing element and interpolate
        for (i, point) in query_points.outer_iter().enumerate() {
            let point_array = [point[0], point[1], point[2]];
            interpolated[i] = self.interpolate_at_point(point_array)?;
        }

        Ok(interpolated)
    }

    /// Interpolate solution at a single point
    fn interpolate_at_point(&self, point: [f64; 3]) -> KwaversResult<f64> {
        // Find element containing the point (simplified - would need proper search)
        if let Some(element) = self.mesh.elements.first() {
            // For simplicity, assume point is in first element
            // In full implementation, would need element search/localization

            // Map physical point to reference coordinates (simplified)
            let xi = 0.0; // Would compute from inverse mapping
            let eta = 0.0; // Would compute from inverse mapping
            let zeta = 0.0; // Would compute from inverse mapping

            // Interpolate using SEM basis functions
            let n_gll = self.mesh.basis.n_points();
            let mut result = 0.0;

            for i in 0..n_gll {
                for j in 0..n_gll {
                    for k in 0..n_gll {
                        let basis_value = self.mesh.basis.lagrange(i, xi)
                            * self.mesh.basis.lagrange(j, eta)
                            * self.mesh.basis.lagrange(k, zeta);

                        let global_dof =
                            self.element_local_to_global_dof(element.id, i, j, k, n_gll);

                        result += basis_value * self.solution[global_dof];
                    }
                }
            }

            Ok(result)
        } else {
            Err(crate::core::error::KwaversError::InvalidInput(
                "No elements in mesh".to_string(),
            ))
        }
    }

    /// Get mesh reference
    #[must_use]
    pub fn mesh(&self) -> &Arc<SemMesh> {
        &self.mesh
    }

    /// Get solver configuration
    #[must_use]
    pub fn config(&self) -> &SemConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::forward::sem::mesh::MeshBuilder;

    #[test]
    fn test_sem_solver_creation() {
        let mesh = MeshBuilder::create_rectangular_mesh(1.0, 1.0, 1.0, 4);
        let config = SemConfig::default();

        let solver = SemSolver::new(config, Arc::new(mesh)).unwrap();

        assert_eq!(solver.config.polynomial_degree, 4); // Default
        assert_eq!(solver.solution.len(), 125); // (4+1)³ = 125 DOFs for degree 4 mesh
    }

    #[test]
    fn test_sem_system_assembly() {
        let mesh = MeshBuilder::create_rectangular_mesh(1.0, 1.0, 1.0, 2);
        let mut config = SemConfig::default();
        config.polynomial_degree = 2;

        let mut solver = SemSolver::new(config, Arc::new(mesh)).unwrap();

        solver.assemble_system().unwrap();

        // Check that mass matrix has been populated
        assert!(solver.mass_matrix.iter().any(|&m| m > 0.0));
    }

    #[test]
    fn test_sem_time_stepping() {
        let mesh = MeshBuilder::create_rectangular_mesh(0.1, 0.1, 0.1, 2);
        let mut config = SemConfig::default();
        config.polynomial_degree = 2;
        config.n_steps = 5;
        config.dt = 1e-8; // Very small time step for stability

        let mut solver = SemSolver::new(config, Arc::new(mesh)).unwrap();
        solver.assemble_system().unwrap();

        // Run a few time steps
        for _ in 0..3 {
            solver.step().unwrap();
        }

        assert_eq!(solver.current_step(), 3);
        assert!(solver.current_time() > 0.0);
    }

    #[test]
    fn test_boundary_condition_management() {
        let mesh = MeshBuilder::create_rectangular_mesh(1.0, 1.0, 1.0, 2);
        let mut config = SemConfig::default();
        config.polynomial_degree = 2;

        let mut solver = SemSolver::new(config, Arc::new(mesh)).unwrap();

        // Add boundary condition
        solver
            .boundary_manager()
            .add_dirichlet(vec![(0, Complex64::new(1.0, 0.0))]);

        assert_eq!(solver.boundary_manager_ref().len(), 1);
    }
}
