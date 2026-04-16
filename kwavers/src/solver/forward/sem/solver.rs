//! Spectral Element Method Solver Implementation
//!
//! Main solver for the Spectral Element Method, integrating:
//! - High-order basis functions on GLL points
//! - Hexahedral element assembly
//! - Time integration (Newmark method)
//! - Boundary condition management
//!
//! ## Stiffness Application — Matrix-Free Sum Factorisation
//!
//! The stiffness operator K is applied matrix-free using the standard SEM
//! sum-factorisation algorithm (Komatitsch & Tromp 1999, §3):
//!
//! ```text
//! For each element e:
//!   1. Spectral derivatives via D-matrix:
//!      d_ξ[i,j,k] = Σ_a D[i,a] u_e[a,j,k]
//!      d_η[i,j,k] = Σ_b D[j,b] u_e[i,b,k]
//!      d_ζ[i,j,k] = Σ_c D[k,c] u_e[i,j,c]
//!
//!   2. Physical metric tensor at each GLL point (p,q,r):
//!      G_αβ[p,q,r] = |J[p,q,r]| × Σ_s J⁻¹[p,q,r,α,s] J⁻¹[p,q,r,β,s]
//!
//!   3. Weighted physical flux (single-direction weight):
//!      q_ξ[i,j,k] = ρc² w_i Σ_β G_1β[i,j,k] d_β[i,j,k]
//!      q_η[i,j,k] = ρc² w_j Σ_β G_2β[i,j,k] d_β[i,j,k]
//!      q_ζ[i,j,k] = ρc² w_k Σ_β G_3β[i,j,k] d_β[i,j,k]
//!
//!   4. Transpose derivative (scatter to DOFs):
//!      (Ku)_{abc} = w_b w_c Σ_i D[i,a] q_ξ[i,b,c]
//!                 + w_a w_c Σ_j D[j,b] q_η[a,j,c]
//!                 + w_a w_b Σ_k D[k,c] q_ζ[a,b,k]
//! ```
//!
//! ## References
//!
//! - Komatitsch D, Tromp J (1999). "Introduction to the spectral element
//!   method for three-dimensional seismic wave propagation."
//!   *Geophys J Int* **139**, 806–822. doi:10.1046/j.1365-246x.1999.00967.x

use crate::core::error::KwaversResult;
use crate::domain::boundary::FemBoundaryManager;
use crate::math::linear_algebra::sparse::CompressedSparseRowMatrix;
use ndarray::{Array1, Array2, Array3};
use std::sync::Arc;

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
    #[allow(dead_code)]
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

    /// Assemble the global diagonal mass matrix.
    ///
    /// The stiffness operator is applied matrix-free via [`Self::apply_stiffness`]
    /// using the sum-factorisation algorithm (Komatitsch & Tromp 1999).
    ///
    /// ## Mass matrix — SEM diagonal lumping
    ///
    /// In the SEM with GLL quadrature, the Lagrange basis satisfies N_i(ξ_j) = δ_{ij},
    /// so the consistent mass matrix is diagonal (no quadrature error):
    ///
    /// ```text
    /// M_ii = ρ ∫ N_i² dΩ = ρ |J_i| w_i w_j w_k   (GLL quadrature; exact for p ≤ 2N-1)
    /// ```
    ///
    /// ## Reference
    ///
    /// Komatitsch & Tromp (1999), Geophys J Int 139 §2.2.
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
                        // M_ii = ρ |J_i| w_i w_j w_k
                        self.mass_matrix[ijk] +=
                            self.config.density * jacobian_det[[i, j, k]] * w;
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

    /// Set initial displacement and synchronise the time integrator.
    ///
    /// Both `self.solution` and the integrator's internal displacement state are
    /// updated so that the first `step()` call correctly advances from u₀ rather
    /// than from the integrator's default zero state.
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
        // Sync integrator so the first Newmark step starts from u₀, not 0.
        self.integrator.set_initial_displacement(&initial_displacement);
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

    /// Apply the stiffness operator K·u matrix-free using 3-D sum-factorisation.
    ///
    /// ## Algorithm (Komatitsch & Tromp 1999, §3)
    ///
    /// For each element *e* with local DOF array u_e[a,b,c]:
    ///
    /// **Step 1 — reference-space gradients** via the spectral derivative matrix
    /// D[p,i] = dℓ_i/dξ|_{ξ_p} = `lagrange_derivatives[[i,p]]`:
    /// ```text
    /// d_ξ[p,q,r] = Σ_a D[p,a] u_e[a,q,r]
    /// d_η[p,q,r] = Σ_b D[q,b] u_e[p,b,r]
    /// d_ζ[p,q,r] = Σ_c D[r,c] u_e[p,q,c]
    /// ```
    ///
    /// **Step 2 — metric tensor** at each GLL point (p,q,r):
    /// ```text
    /// G_{αβ}[p,q,r] = Σ_m J⁻¹[p,q,r,α,m] J⁻¹[p,q,r,β,m]
    /// ```
    ///
    /// **Step 3 — weighted physical flux**:
    /// ```text
    /// q_ξ[p,q,r] = ρc² |J[p,q,r]| w_p  (G·∇_ξ u)[0]
    /// q_η[p,q,r] = ρc² |J[p,q,r]| w_q  (G·∇_ξ u)[1]
    /// q_ζ[p,q,r] = ρc² |J[p,q,r]| w_r  (G·∇_ξ u)[2]
    /// ```
    ///
    /// **Step 4 — transpose derivative (scatter)**:
    /// ```text
    /// (Ku)_{abc} = w_b w_c Σ_p D[p,a] q_ξ[p,b,c]
    ///            + w_a w_c Σ_q D[q,b] q_η[a,q,c]
    ///            + w_a w_b Σ_r D[r,c] q_ζ[a,b,r]
    /// ```
    fn apply_stiffness(&self, u: &Array1<f64>) -> Array1<f64> {
        let mut ku = Array1::<f64>::zeros(u.len());
        let n = self.mesh.basis.n_points();
        // ld[[i,p]] = dℓ_i/dξ|_{ξ_p}  =>  D[p,i] = ld[[i,p]]
        let ld = &self.mesh.basis.lagrange_derivatives;
        let w = &self.mesh.basis.gll_weights;
        let rho_c2 = self.config.density * self.config.sound_speed * self.config.sound_speed;

        for elem_idx in 0..self.mesh.elements.len() {
            let element = &self.mesh.elements[elem_idx];
            let eid = element.id;

            // Step 1: Gather u_e[a,b,c] from global DOFs
            let mut u_e = Array3::<f64>::zeros((n, n, n));
            for a in 0..n {
                for b in 0..n {
                    for c in 0..n {
                        let g = self.element_local_to_global_dof(eid, a, b, c, n);
                        u_e[[a, b, c]] = u[g];
                    }
                }
            }

            // Step 2: Reference-space gradients
            // d_xi[p,q,r]  = Σ_a ld[[a,p]] * u_e[a,q,r]   (D[p,a] = ld[[a,p]])
            // d_eta[p,q,r] = Σ_b ld[[b,q]] * u_e[p,b,r]
            // d_zeta[p,q,r]= Σ_c ld[[c,r]] * u_e[p,q,c]
            let mut d_xi = Array3::<f64>::zeros((n, n, n));
            let mut d_eta = Array3::<f64>::zeros((n, n, n));
            let mut d_zeta = Array3::<f64>::zeros((n, n, n));

            for p in 0..n {
                for q in 0..n {
                    for r in 0..n {
                        let mut dxi = 0.0;
                        let mut deta = 0.0;
                        let mut dzeta = 0.0;
                        for a in 0..n {
                            dxi += ld[[a, p]] * u_e[[a, q, r]];
                            deta += ld[[a, q]] * u_e[[p, a, r]];
                            dzeta += ld[[a, r]] * u_e[[p, q, a]];
                        }
                        d_xi[[p, q, r]] = dxi;
                        d_eta[[p, q, r]] = deta;
                        d_zeta[[p, q, r]] = dzeta;
                    }
                }
            }

            // Steps 3–4: Metric tensor and weighted fluxes
            // G_{αβ} = Σ_m Jinv[α,m] Jinv[β,m]
            // (G·d)[α] = Σ_β G_{αβ} d_β = Σ_m Jinv[α,m] (Σ_β Jinv[β,m] d_β)
            let mut q_xi = Array3::<f64>::zeros((n, n, n));
            let mut q_eta = Array3::<f64>::zeros((n, n, n));
            let mut q_zeta = Array3::<f64>::zeros((n, n, n));

            for p in 0..n {
                for q in 0..n {
                    for r in 0..n {
                        let jdet = element.jacobian_det[[p, q, r]];
                        let jinv = element.jacobian_inv.slice(ndarray::s![p, q, r, .., ..]);
                        // jinv[[α, m]] = J⁻¹[α, m] = dξ_α/dx_m

                        let d = [d_xi[[p, q, r]], d_eta[[p, q, r]], d_zeta[[p, q, r]]];

                        // h_m = Σ_β Jinv[β,m] d_β   (inner product of each column of Jinv with d)
                        let mut h = [0.0f64; 3];
                        for beta in 0..3usize {
                            for m in 0..3usize {
                                h[m] += jinv[[beta, m]] * d[beta];
                            }
                        }
                        // (G·d)[α] = Σ_m Jinv[α,m] h_m
                        let mut gd = [0.0f64; 3];
                        for alpha in 0..3usize {
                            for m in 0..3usize {
                                gd[alpha] += jinv[[alpha, m]] * h[m];
                            }
                        }

                        let scale = rho_c2 * jdet;
                        q_xi[[p, q, r]] = scale * w[p] * gd[0];
                        q_eta[[p, q, r]] = scale * w[q] * gd[1];
                        q_zeta[[p, q, r]] = scale * w[r] * gd[2];
                    }
                }
            }

            // Step 5: Transpose derivative — scatter weighted fluxes to DOFs
            // (Ku)_{abc} += w_b w_c Σ_p ld[[a,p]] q_xi[p,b,c]
            //             + w_a w_c Σ_q ld[[b,q]] q_eta[a,q,c]
            //             + w_a w_b Σ_r ld[[c,r]] q_zeta[a,b,r]
            for a in 0..n {
                for b in 0..n {
                    for c in 0..n {
                        let mut xi_sum = 0.0;
                        for p in 0..n {
                            xi_sum += ld[[a, p]] * q_xi[[p, b, c]];
                        }
                        let mut eta_sum = 0.0;
                        for q in 0..n {
                            eta_sum += ld[[b, q]] * q_eta[[a, q, c]];
                        }
                        let mut zeta_sum = 0.0;
                        for r in 0..n {
                            zeta_sum += ld[[c, r]] * q_zeta[[a, b, r]];
                        }

                        let val = w[b] * w[c] * xi_sum
                            + w[a] * w[c] * eta_sum
                            + w[a] * w[b] * zeta_sum;

                        let g = self.element_local_to_global_dof(eid, a, b, c, n);
                        ku[g] += val;
                    }
                }
            }
        }

        ku
    }

    /// Compute nodal acceleration from the equation of motion M·ü + K·u = 0.
    ///
    /// The SEM mass matrix is diagonal (GLL mass lumping), so:
    /// ```text
    /// ü_i = −(K·u)_i / M_ii
    /// ```
    ///
    /// K·u is computed matrix-free by [`Self::apply_stiffness`].
    fn compute_acceleration(&self) -> KwaversResult<Array1<f64>> {
        let ku = self.apply_stiffness(&self.solution);
        let n_dofs = self.solution.len();
        let mut acceleration = Array1::<f64>::zeros(n_dofs);

        for i in 0..n_dofs {
            let m = self.mass_matrix[i];
            if !m.is_finite() || m <= 0.0 {
                return Err(crate::core::error::KwaversError::InvalidInput(format!(
                    "Non-positive mass matrix entry at dof {i}: {m}"
                )));
            }
            acceleration[i] = -ku[i] / m;
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
    fn interpolate_at_point(&self, _point: [f64; 3]) -> KwaversResult<f64> {
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
    use num_complex::Complex64;

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
        let config = SemConfig {
            polynomial_degree: 2,
            ..Default::default()
        };

        let mut solver = SemSolver::new(config, Arc::new(mesh)).unwrap();

        solver.assemble_system().unwrap();

        // Check that mass matrix has been populated
        assert!(solver.mass_matrix.iter().any(|&m| m > 0.0));
    }

    #[test]
    fn test_sem_time_stepping() {
        let mesh = MeshBuilder::create_rectangular_mesh(0.1, 0.1, 0.1, 2);
        let config = SemConfig {
            polynomial_degree: 2,
            n_steps: 5,
            dt: 1e-8,
            ..Default::default()
        };

        let mut solver = SemSolver::new(config, Arc::new(mesh)).unwrap();
        solver.assemble_system().unwrap();

        // Run a few time steps
        for _ in 0..3 {
            solver.step().unwrap();
        }

        assert_eq!(solver.current_step(), 3);
        assert!(solver.current_time() > 0.0);
    }

    /// Constant field u=1: ∇u=0 everywhere, so K·u must be identically zero.
    /// We use a relative tolerance vs the stiffness scale ρc² to account for
    /// floating-point summation with O(N³) GLL quadrature points.
    #[test]
    fn test_stiffness_constant_field_is_zero() {
        let mesh = MeshBuilder::create_rectangular_mesh(1.0, 1.0, 1.0, 3);
        let config = SemConfig {
            polynomial_degree: 3,
            sound_speed: 1500.0,
            density: 1000.0,
            ..Default::default()
        };
        let mut solver = SemSolver::new(config, Arc::new(mesh)).unwrap();
        solver.assemble_system().unwrap();

        let n = solver.solution.len();
        let u_const = Array1::<f64>::ones(n);
        let ku = solver.apply_stiffness(&u_const);

        // Scale: ρc² = 2.25e9; allow relative tolerance of 1e-10 vs this scale.
        let scale = solver.config.density * solver.config.sound_speed.powi(2);
        let tol = scale * 1e-10;
        for (i, &v) in ku.iter().enumerate() {
            assert!(
                v.abs() < tol,
                "K·1 should be zero at dof {i}, got {v:.3e} (tol {tol:.3e})"
            );
        }
    }

    /// Stiffness energy for u=x on [0,Lx]×[0,Ly]×[0,Lz]:
    ///
    /// uᵀKu = ρc² ∫|∇u|² dΩ = ρc² ∫ 1² dxdydz = ρc² × Lx × Ly × Lz
    ///
    /// (Integration by parts shows K·x ≠ 0 at boundary nodes due to the
    /// natural Neumann flux; we test the energy integral which has the
    /// clean analytical form.)
    #[test]
    fn test_stiffness_energy_linear_field() {
        let lx = 2.0;
        let ly = 1.5;
        let lz = 1.0;
        let mesh = MeshBuilder::create_rectangular_mesh(lx, ly, lz, 3);
        let config = SemConfig {
            polynomial_degree: 3,
            sound_speed: 1500.0,
            density: 1000.0,
            ..Default::default()
        };
        let mut solver = SemSolver::new(config, Arc::new(mesh)).unwrap();
        solver.assemble_system().unwrap();

        let n = solver.mesh.basis.n_points();
        let mut u = Array1::<f64>::zeros(solver.solution.len());
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    let xi = solver.mesh.basis.gll_points[a];
                    let x = lx * (xi + 1.0) / 2.0;
                    let g = solver.element_local_to_global_dof(0, a, b, c, n);
                    u[g] = x;
                }
            }
        }

        let ku = solver.apply_stiffness(&u);
        let energy: f64 = u.iter().zip(ku.iter()).map(|(ui, kui)| ui * kui).sum();

        // ρc² × Lx × Ly × Lz = 1000 × 1500² × 2.0 × 1.5 × 1.0 = 6.75e12
        let expected = solver.config.density
            * solver.config.sound_speed.powi(2)
            * lx * ly * lz;
        let rel_err = (energy - expected).abs() / expected;
        assert!(
            rel_err < 1e-6,
            "Stiffness energy u^T K u = {energy:.6e}, expected {expected:.6e} (rel err {rel_err:.2e})"
        );
    }

    /// Symmetry test: K is symmetric, so (K·u)·v = u·(K·v) for all u,v.
    #[test]
    fn test_stiffness_symmetry() {
        let mesh = MeshBuilder::create_rectangular_mesh(1.0, 1.0, 1.0, 2);
        let config = SemConfig {
            polynomial_degree: 2,
            ..Default::default()
        };
        let mut solver = SemSolver::new(config, Arc::new(mesh)).unwrap();
        solver.assemble_system().unwrap();

        let n = solver.solution.len();
        // u = sin(π i / n), v = cos(π i / n)
        let u: Array1<f64> =
            (0..n).map(|i| (std::f64::consts::PI * i as f64 / n as f64).sin()).collect();
        let v: Array1<f64> =
            (0..n).map(|i| (std::f64::consts::PI * i as f64 / n as f64).cos()).collect();

        let ku = solver.apply_stiffness(&u);
        let kv = solver.apply_stiffness(&v);

        let ku_dot_v: f64 = ku.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        let u_dot_kv: f64 = u.iter().zip(kv.iter()).map(|(a, b)| a * b).sum();

        let scale = ku_dot_v.abs().max(u_dot_kv.abs()).max(1e-30);
        assert!(
            (ku_dot_v - u_dot_kv).abs() / scale < 1e-10,
            "Stiffness asymmetry: (Ku)·v={ku_dot_v:.6e} u·(Kv)={u_dot_kv:.6e}"
        );
    }

    /// Free vibration energy conservation: E = ½(vᵀMv + uᵀKu) should be
    /// constant (within time-integration truncation error) over multiple steps.
    #[test]
    fn test_free_vibration_energy_conservation() {
        let mesh = MeshBuilder::create_rectangular_mesh(0.1, 0.1, 0.1, 2);
        let config = SemConfig {
            polynomial_degree: 2,
            n_steps: 20,
            dt: 1e-9,
            sound_speed: 1500.0,
            density: 1000.0,
            wavenumber: 1.0,
        };
        let mut solver = SemSolver::new(config, Arc::new(mesh)).unwrap();
        solver.assemble_system().unwrap();

        // Non-trivial initial displacement: u_a = sin(π a / N) on first GLL index
        let n = solver.mesh.basis.n_points();
        let n_dofs = solver.solution.len();
        let mut u0 = Array1::<f64>::zeros(n_dofs);
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    let g = solver.element_local_to_global_dof(0, a, b, c, n);
                    u0[g] = (std::f64::consts::PI * a as f64 / (n - 1).max(1) as f64).sin();
                }
            }
        }
        solver.set_initial_conditions(u0).unwrap();

        let compute_energy = |sol: &SemSolver| -> f64 {
            let ku = sol.apply_stiffness(&sol.solution);
            let v = &sol.integrator.velocity;
            let potential: f64 = sol.solution.iter().zip(ku.iter()).map(|(u, ku)| u * ku).sum();
            let kinetic: f64 = v
                .iter()
                .zip(sol.mass_matrix.iter())
                .map(|(vi, mi)| vi * vi * mi)
                .sum();
            0.5 * (kinetic + potential)
        };

        let e0 = compute_energy(&solver);
        // Guard against degenerate zero-energy start
        if e0 < 1e-30 {
            return;
        }

        for _ in 0..20 {
            solver.step().unwrap();
        }

        let e_final = compute_energy(&solver);
        // Newmark average-acceleration is unconditionally stable but not
        // exactly energy-conserving; allow 1% relative drift over 20 steps.
        let relative_drift = (e_final - e0).abs() / e0;
        assert!(
            relative_drift < 0.01,
            "Energy drift {relative_drift:.3e} exceeds 1% over 20 steps (E0={e0:.3e}, Ef={e_final:.3e})"
        );
    }

    #[test]
    fn test_boundary_condition_management() {
        let mesh = MeshBuilder::create_rectangular_mesh(1.0, 1.0, 1.0, 2);
        let config = SemConfig {
            polynomial_degree: 2,
            ..Default::default()
        };

        let mut solver = SemSolver::new(config, Arc::new(mesh)).unwrap();

        // Add boundary condition
        solver
            .boundary_manager()
            .add_dirichlet(vec![(0, Complex64::new(1.0, 0.0))]);

        assert_eq!(solver.boundary_manager_ref().len(), 1);
    }
}
