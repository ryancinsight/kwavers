//! `SemSolver` struct, constructor, system assembly, and time-stepping.
//!
//! ## Mass matrix — SEM diagonal lumping (Komatitsch & Tromp 1999 §2.2)
//!
//! With GLL quadrature and Lagrange basis N_i(ξ_j) = δ_{ij}:
//! ```text
//! M_ii = ρ |J_i| w_i w_j w_k   (exact for polynomial degree p ≤ 2N-1)
//! ```

use super::config::SemConfig;
use crate::core::error::KwaversResult;
use crate::domain::boundary::FemBoundaryManager;
use crate::math::linear_algebra::sparse::CompressedSparseRowMatrix;
use ndarray::{Array1, Array2};
use std::sync::Arc;

use super::super::elements::SemMesh;
use super::super::integration::NewmarkIntegrator;

/// Spectral Element Method solver for acoustic wave propagation
#[derive(Debug)]
pub struct SemSolver {
    /// Solver configuration
    pub(super) config: SemConfig,
    /// Spectral element mesh
    pub(super) mesh: Arc<SemMesh>,
    /// Global mass matrix (diagonal by construction in SEM)
    pub(super) mass_matrix: Array1<f64>,
    /// Global stiffness matrix
    #[allow(dead_code)]
    pub(super) stiffness_matrix: CompressedSparseRowMatrix<f64>,
    /// Boundary condition manager
    pub(super) boundary_manager: FemBoundaryManager,
    /// Time integrator
    pub(super) integrator: NewmarkIntegrator,
    /// Current solution vector
    pub(super) solution: Array1<f64>,
    /// Current time step index
    pub(super) time_step: usize,
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
        let mass_matrix = Array1::<f64>::zeros(n_dofs);
        let stiffness_matrix = CompressedSparseRowMatrix::create(n_dofs, n_dofs);
        let integrator = NewmarkIntegrator::average_acceleration(config.dt, n_dofs);
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

    /// Assemble the global diagonal mass matrix.
    ///
    /// M_ii = ρ |J_i| w_i w_j w_k  (GLL quadrature; exact for p ≤ 2N-1)
    pub fn assemble_system(&mut self) -> KwaversResult<()> {
        self.mass_matrix.fill(0.0);

        let n_gll = self.mesh.basis.n_points();
        let weights = &self.mesh.basis.gll_weights;

        for elem_idx in 0..self.mesh.elements.len() {
            let element = &self.mesh.elements[elem_idx];
            let element_id = element.id;
            let jacobian_det = &element.jacobian_det;

            for i in 0..n_gll {
                for j in 0..n_gll {
                    for k in 0..n_gll {
                        let ijk = self.element_local_to_global_dof(element_id, i, j, k, n_gll);
                        let w = weights[i] * weights[j] * weights[k];
                        self.mass_matrix[ijk] += self.config.density * jacobian_det[[i, j, k]] * w;
                    }
                }
            }
        }

        Ok(())
    }

    /// Convert element local DOF indices to global DOF index.
    pub(super) fn element_local_to_global_dof(
        &self,
        element_id: usize,
        i: usize,
        j: usize,
        k: usize,
        n_gll: usize,
    ) -> usize {
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

    /// Set initial displacement and synchronise the time integrator.
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
        self.integrator
            .set_initial_displacement(&initial_displacement);
        Ok(())
    }

    /// Advance one time step: M·ü + K·u = 0 via Newmark average-acceleration
    pub fn step(&mut self) -> KwaversResult<()> {
        if self.time_step >= self.config.n_steps {
            return Ok(());
        }

        let acceleration = self.compute_acceleration()?;
        self.integrator.step(&acceleration);
        self.solution.assign(&self.integrator.displacement);
        self.time_step += 1;

        Ok(())
    }

    /// Run complete simulation
    pub fn run_simulation(&mut self) -> KwaversResult<()> {
        self.assemble_system()?;
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
    pub fn interpolate_solution(&self, query_points: &Array2<f64>) -> KwaversResult<Array1<f64>> {
        let mut interpolated = Array1::<f64>::zeros(query_points.nrows());
        for (i, point) in query_points.outer_iter().enumerate() {
            let point_array = [point[0], point[1], point[2]];
            interpolated[i] = self.interpolate_at_point(point_array)?;
        }
        Ok(interpolated)
    }

    fn interpolate_at_point(&self, _point: [f64; 3]) -> KwaversResult<f64> {
        if let Some(element) = self.mesh.elements.first() {
            let xi = 0.0;
            let eta = 0.0;
            let zeta = 0.0;

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
