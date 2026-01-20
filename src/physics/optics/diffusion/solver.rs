//! Diffusion Approximation Solver for Optical Fluence Computation
//!
//! # Mathematical Foundation
//!
//! ## Steady-State Diffusion Equation
//!
//! For continuous-wave (CW) illumination, the photon fluence Φ(r) satisfies:
//!
//! ```text
//! ∇·(D(r)∇Φ(r)) - μₐ(r)Φ(r) = -S(r)
//! ```
//!
//! Where:
//! - `Φ(r)`: Optical fluence (W/m²)
//! - `D(r) = 1/(3(μₐ + μₛ'))`: Diffusion coefficient (m)
//! - `μₐ(r)`: Absorption coefficient (m⁻¹)
//! - `μₛ'(r) = μₛ(1-g)`: Reduced scattering coefficient (m⁻¹)
//! - `S(r)`: Isotropic source term (W/m³)
//!
//! ## Boundary Conditions (Extrapolated Boundary)
//!
//! At tissue-air interface, partial current boundary condition:
//!
//! ```text
//! Φ(r_b) + 2A D(r_b) ∂Φ/∂n|_{r_b} = 0
//! ```
//!
//! Where `A = (1 + R_eff)/(1 - R_eff)` accounts for internal reflection.
//! For typical tissue-air interface (n=1.4), `A ≈ 2.0`.
//!
//! ## Discretization (Finite Difference Method)
//!
//! Second-order central differences on uniform Cartesian grid:
//!
//! ```text
//! ∇·(D∇Φ) ≈ (D_{i+1/2}(Φ_{i+1} - Φᵢ) - D_{i-1/2}(Φᵢ - Φ_{i-1}))/Δx²
//! ```
//!
//! Results in 7-point stencil for 3D (19-point for heterogeneous D).
//!
//! ## References
//!
//! - **Arridge (1999)**: "Optical tomography in medical imaging." *Inverse Problems*
//! - **Wang & Jacques (1995)**: "Monte Carlo modeling of light transport." *Computer Methods*
//! - **Contini et al. (1997)**: "Photon migration through a turbid slab." *Applied Optics*

use crate::domain::grid::Grid;
use crate::domain::medium::properties::OpticalPropertyData;
use anyhow::Result;
use ndarray::Array3;

/// Configuration for diffusion solver
#[derive(Debug, Clone)]
pub struct DiffusionSolverConfig {
    /// Maximum number of conjugate gradient iterations
    pub max_iterations: usize,
    /// Convergence tolerance (relative residual)
    pub tolerance: f64,
    /// Extrapolated boundary parameter A (default 2.0 for tissue-air)
    pub boundary_parameter: f64,
    pub boundary_conditions: Option<DiffusionBoundaryConditions>,
    /// Enable verbose convergence logging
    pub verbose: bool,
}

impl Default for DiffusionSolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10000,
            tolerance: 1e-6,
            boundary_parameter: 2.0,
            boundary_conditions: None,
            verbose: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DiffusionBoundaryCondition {
    ZeroFlux,
    Extrapolated { a: f64 },
}

#[derive(Debug, Clone, Copy)]
pub struct DiffusionBoundaryConditions {
    pub x_min: DiffusionBoundaryCondition,
    pub x_max: DiffusionBoundaryCondition,
    pub y_min: DiffusionBoundaryCondition,
    pub y_max: DiffusionBoundaryCondition,
    pub z_min: DiffusionBoundaryCondition,
    pub z_max: DiffusionBoundaryCondition,
}

impl DiffusionBoundaryConditions {
    #[must_use]
    pub fn all_extrapolated(a: f64) -> Self {
        Self {
            x_min: DiffusionBoundaryCondition::Extrapolated { a },
            x_max: DiffusionBoundaryCondition::Extrapolated { a },
            y_min: DiffusionBoundaryCondition::Extrapolated { a },
            y_max: DiffusionBoundaryCondition::Extrapolated { a },
            z_min: DiffusionBoundaryCondition::Extrapolated { a },
            z_max: DiffusionBoundaryCondition::Extrapolated { a },
        }
    }
}

impl Default for DiffusionBoundaryConditions {
    fn default() -> Self {
        Self::all_extrapolated(2.0)
    }
}

/// Steady-state diffusion solver for optical fluence
///
/// Solves: ∇·(D(r)∇Φ(r)) - μₐ(r)Φ(r) = -S(r)
/// using finite difference discretization and conjugate gradient iteration.
#[derive(Debug)]
pub struct DiffusionSolver {
    /// Computational grid
    grid: Grid,
    /// Spatial diffusion coefficient field D(r) = 1/(3(μₐ + μₛ'))
    diffusion_coefficient: Array3<f64>,
    /// Absorption coefficient field μₐ(r)
    absorption_coefficient: Array3<f64>,
    /// Solver configuration
    config: DiffusionSolverConfig,
}

impl DiffusionSolver {
    fn boundary_conditions(&self) -> DiffusionBoundaryConditions {
        self.config.boundary_conditions.unwrap_or_else(|| {
            DiffusionBoundaryConditions::all_extrapolated(self.config.boundary_parameter)
        })
    }

    fn ghost_coefficient(
        boundary_condition: DiffusionBoundaryCondition,
        diffusion_coefficient: f64,
        delta: f64,
    ) -> f64 {
        match boundary_condition {
            DiffusionBoundaryCondition::ZeroFlux => 1.0,
            DiffusionBoundaryCondition::Extrapolated { a } => {
                let r = 4.0 * a * diffusion_coefficient / delta;
                if r <= 0.0 {
                    0.0
                } else if (r + 1.0).abs() > 1e-30 {
                    (r - 1.0) / (r + 1.0)
                } else {
                    0.0
                }
            }
        }
    }

    /// Create solver from spatially-varying optical property map
    ///
    /// # Arguments
    ///
    /// - `grid`: Computational domain discretization
    /// - `optical_properties`: Spatial map of optical properties (from domain SSOT)
    /// - `config`: Solver configuration
    ///
    /// # Example
    ///
    /// ```no_run
    /// use kwavers::domain::grid::Grid;
    /// use kwavers::domain::medium::properties::OpticalPropertyData;
    /// use kwavers::physics::optics::diffusion::solver::{DiffusionSolver, DiffusionSolverConfig};
    /// use ndarray::Array3;
    ///
    /// let grid = Grid::new(50, 50, 50, 1e-3, 1e-3, 1e-3).unwrap();
    /// let tissue = OpticalPropertyData::soft_tissue();
    /// let optical_map = Array3::from_elem(grid.dimensions(), tissue);
    /// let config = DiffusionSolverConfig::default();
    ///
    /// let solver = DiffusionSolver::new(grid, optical_map, config).unwrap();
    /// ```
    pub fn new(
        grid: Grid,
        optical_properties: Array3<OpticalPropertyData>,
        config: DiffusionSolverConfig,
    ) -> Result<Self> {
        let (nx, ny, nz) = grid.dimensions();

        // Validate dimensions
        if optical_properties.dim() != (nx, ny, nz) {
            anyhow::bail!(
                "Optical property map dimensions {:?} do not match grid dimensions ({}, {}, {})",
                optical_properties.shape(),
                nx,
                ny,
                nz
            );
        }

        // Pre-compute diffusion and absorption coefficient fields
        let mut diffusion_coefficient = Array3::zeros((nx, ny, nz));
        let mut absorption_coefficient = Array3::zeros((nx, ny, nz));

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let props = &optical_properties[[i, j, k]];

                    // Diffusion coefficient: D = 1/(3(μₐ + μₛ'))
                    let mu_a = props.absorption_coefficient;
                    let mu_s_prime = props.reduced_scattering();
                    let d_val = 1.0 / (3.0 * (mu_a + mu_s_prime));

                    diffusion_coefficient[[i, j, k]] = d_val;
                    absorption_coefficient[[i, j, k]] = mu_a;
                }
            }
        }

        Ok(Self {
            grid,
            diffusion_coefficient,
            absorption_coefficient,
            config,
        })
    }

    /// Create solver with uniform optical properties (homogeneous medium)
    ///
    /// Convenience constructor for simple scenarios.
    pub fn uniform(
        grid: Grid,
        optical_properties: OpticalPropertyData,
        config: DiffusionSolverConfig,
    ) -> Result<Self> {
        let (nx, ny, nz) = grid.dimensions();
        let optical_map = Array3::from_elem((nx, ny, nz), optical_properties);
        Self::new(grid, optical_map, config)
    }

    /// Solve steady-state diffusion equation for given source distribution
    ///
    /// # Arguments
    ///
    /// - `source`: Isotropic source term S(r) in W/m³
    ///
    /// # Returns
    ///
    /// Optical fluence field Φ(r) in W/m²
    ///
    /// # Algorithm
    ///
    /// Uses preconditioned conjugate gradient (PCG) with Jacobi preconditioner:
    /// 1. Discretize PDE into linear system Ax = b
    /// 2. Iterate: x_{k+1} = x_k + α_k p_k until ||r_k|| < tol
    /// 3. Apply extrapolated boundary conditions at domain boundaries
    pub fn solve(&self, source: &Array3<f64>) -> Result<Array3<f64>> {
        let (nx, ny, nz) = self.grid.dimensions();

        // Validate source dimensions
        if source.dim() != (nx, ny, nz) {
            anyhow::bail!(
                "Source dimensions {:?} do not match grid dimensions ({}, {}, {})",
                source.shape(),
                nx,
                ny,
                nz
            );
        }

        // Initial guess: zero fluence
        let mut fluence = Array3::zeros((nx, ny, nz));

        // Residual: r = b - Ax
        let mut residual = source.clone();

        // Search direction
        let mut search_direction = Array3::zeros((nx, ny, nz));

        // Preconditioner (Jacobi: diagonal elements of A)
        let preconditioner = self.compute_preconditioner();

        let mut preconditioned_residual = &residual * &preconditioner;

        // Initial search direction: p_0 = z_0
        search_direction.assign(&preconditioned_residual);

        // Initial residual norm
        let mut residual_dot_z = (&residual * &preconditioned_residual).sum();
        let initial_residual_norm = residual_dot_z.sqrt();

        if self.config.verbose {
            tracing::info!(
                "DiffusionSolver: Initial residual = {:.6e}",
                initial_residual_norm
            );
        }

        // Conjugate gradient iteration
        for iter in 0..self.config.max_iterations {
            // Matrix-vector product: Ap
            let a_times_p = self.apply_operator(&search_direction);

            // Step size: α = (r^T z) / (p^T Ap)
            let p_dot_ap = (&search_direction * &a_times_p).sum();

            if p_dot_ap.abs() < 1e-30 {
                if self.config.verbose {
                    tracing::warn!(
                        "DiffusionSolver: Near-zero denominator at iteration {}",
                        iter
                    );
                }
                break;
            }

            let alpha = residual_dot_z / p_dot_ap;

            // Update solution: x = x + α p
            fluence = &fluence + &(&search_direction * alpha);

            // Update residual: r = r - α Ap
            residual = &residual - &(&a_times_p * alpha);

            // Check convergence
            let residual_norm = residual.iter().map(|x| x * x).sum::<f64>().sqrt();
            let relative_residual = residual_norm / (initial_residual_norm + 1e-30);

            if self.config.verbose && iter % 100 == 0 {
                tracing::debug!(
                    "DiffusionSolver: Iteration {}, relative residual = {:.6e}",
                    iter,
                    relative_residual
                );
            }

            if relative_residual < self.config.tolerance {
                if self.config.verbose {
                    tracing::info!(
                        "DiffusionSolver: Converged in {} iterations (residual = {:.6e})",
                        iter + 1,
                        relative_residual
                    );
                }
                return Ok(fluence);
            }

            preconditioned_residual = &residual * &preconditioner;

            // Conjugate direction update
            let residual_dot_z_new = (&residual * &preconditioned_residual).sum();
            let beta = residual_dot_z_new / residual_dot_z;
            residual_dot_z = residual_dot_z_new;

            // Update search direction: p = z + β p
            search_direction = &preconditioned_residual + &(&search_direction * beta);
        }

        anyhow::bail!(
            "DiffusionSolver: Failed to converge in {} iterations",
            self.config.max_iterations
        )
    }

    /// Apply diffusion operator: A Φ = ∇·(D∇Φ) - μₐΦ
    ///
    /// Uses second-order finite differences with extrapolated boundary conditions.
    fn apply_operator(&self, fluence: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut result = Array3::zeros((nx, ny, nz));

        let bc = self.boundary_conditions();
        let dx2_inv = 1.0 / (self.grid.dx * self.grid.dx);
        let dy2_inv = 1.0 / (self.grid.dy * self.grid.dy);
        let dz2_inv = 1.0 / (self.grid.dz * self.grid.dz);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let phi_center = fluence[[i, j, k]];
                    let d_center = self.diffusion_coefficient[[i, j, k]];
                    let mu_a = self.absorption_coefficient[[i, j, k]];

                    let mut laplacian = 0.0;

                    // X-direction
                    if nx >= 2 {
                        if i == 0 {
                            let ghost_coeff =
                                Self::ghost_coefficient(bc.x_min, d_center, self.grid.dx);
                            let d_plus =
                                0.5 * (d_center + self.diffusion_coefficient[[i + 1, j, k]]);
                            let phi_plus = fluence[[i + 1, j, k]];
                            let phi_minus = ghost_coeff * phi_center;
                            laplacian += (d_plus * (phi_plus - phi_center)
                                - d_center * (phi_center - phi_minus))
                                * dx2_inv;
                        } else if i + 1 == nx {
                            let ghost_coeff =
                                Self::ghost_coefficient(bc.x_max, d_center, self.grid.dx);
                            let d_minus =
                                0.5 * (d_center + self.diffusion_coefficient[[i - 1, j, k]]);
                            let phi_minus = fluence[[i - 1, j, k]];
                            let phi_plus = ghost_coeff * phi_center;
                            laplacian += (d_center * (phi_plus - phi_center)
                                - d_minus * (phi_center - phi_minus))
                                * dx2_inv;
                        } else {
                            let d_minus =
                                0.5 * (d_center + self.diffusion_coefficient[[i - 1, j, k]]);
                            let phi_minus = fluence[[i - 1, j, k]];
                            let d_plus =
                                0.5 * (d_center + self.diffusion_coefficient[[i + 1, j, k]]);
                            let phi_plus = fluence[[i + 1, j, k]];
                            laplacian += (d_plus * (phi_plus - phi_center)
                                - d_minus * (phi_center - phi_minus))
                                * dx2_inv;
                        }
                    }

                    // Y-direction
                    if ny >= 2 {
                        if j == 0 {
                            let ghost_coeff =
                                Self::ghost_coefficient(bc.y_min, d_center, self.grid.dy);
                            let d_plus =
                                0.5 * (d_center + self.diffusion_coefficient[[i, j + 1, k]]);
                            let phi_plus = fluence[[i, j + 1, k]];
                            let phi_minus = ghost_coeff * phi_center;
                            laplacian += (d_plus * (phi_plus - phi_center)
                                - d_center * (phi_center - phi_minus))
                                * dy2_inv;
                        } else if j + 1 == ny {
                            let ghost_coeff =
                                Self::ghost_coefficient(bc.y_max, d_center, self.grid.dy);
                            let d_minus =
                                0.5 * (d_center + self.diffusion_coefficient[[i, j - 1, k]]);
                            let phi_minus = fluence[[i, j - 1, k]];
                            let phi_plus = ghost_coeff * phi_center;
                            laplacian += (d_center * (phi_plus - phi_center)
                                - d_minus * (phi_center - phi_minus))
                                * dy2_inv;
                        } else {
                            let d_minus =
                                0.5 * (d_center + self.diffusion_coefficient[[i, j - 1, k]]);
                            let phi_minus = fluence[[i, j - 1, k]];
                            let d_plus =
                                0.5 * (d_center + self.diffusion_coefficient[[i, j + 1, k]]);
                            let phi_plus = fluence[[i, j + 1, k]];
                            laplacian += (d_plus * (phi_plus - phi_center)
                                - d_minus * (phi_center - phi_minus))
                                * dy2_inv;
                        }
                    }

                    // Z-direction
                    if nz >= 2 {
                        if k == 0 {
                            let ghost_coeff =
                                Self::ghost_coefficient(bc.z_min, d_center, self.grid.dz);
                            let d_plus =
                                0.5 * (d_center + self.diffusion_coefficient[[i, j, k + 1]]);
                            let phi_plus = fluence[[i, j, k + 1]];
                            let phi_minus = ghost_coeff * phi_center;
                            laplacian += (d_plus * (phi_plus - phi_center)
                                - d_center * (phi_center - phi_minus))
                                * dz2_inv;
                        } else if k + 1 == nz {
                            let ghost_coeff =
                                Self::ghost_coefficient(bc.z_max, d_center, self.grid.dz);
                            let d_minus =
                                0.5 * (d_center + self.diffusion_coefficient[[i, j, k - 1]]);
                            let phi_minus = fluence[[i, j, k - 1]];
                            let phi_plus = ghost_coeff * phi_center;
                            laplacian += (d_center * (phi_plus - phi_center)
                                - d_minus * (phi_center - phi_minus))
                                * dz2_inv;
                        } else {
                            let d_minus =
                                0.5 * (d_center + self.diffusion_coefficient[[i, j, k - 1]]);
                            let phi_minus = fluence[[i, j, k - 1]];
                            let d_plus =
                                0.5 * (d_center + self.diffusion_coefficient[[i, j, k + 1]]);
                            let phi_plus = fluence[[i, j, k + 1]];
                            laplacian += (d_plus * (phi_plus - phi_center)
                                - d_minus * (phi_center - phi_minus))
                                * dz2_inv;
                        }
                    }

                    result[[i, j, k]] = -laplacian + mu_a * phi_center;
                }
            }
        }

        result
    }

    /// Compute Jacobi preconditioner (diagonal of system matrix)
    fn compute_preconditioner(&self) -> Array3<f64> {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut preconditioner = Array3::zeros((nx, ny, nz));

        let bc = self.boundary_conditions();
        let dx2_inv = 1.0 / (self.grid.dx * self.grid.dx);
        let dy2_inv = 1.0 / (self.grid.dy * self.grid.dy);
        let dz2_inv = 1.0 / (self.grid.dz * self.grid.dz);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let d = self.diffusion_coefficient[[i, j, k]];
                    let mu_a = self.absorption_coefficient[[i, j, k]];

                    let ghost_coeff_x_min = Self::ghost_coefficient(bc.x_min, d, self.grid.dx);
                    let ghost_coeff_x_max = Self::ghost_coefficient(bc.x_max, d, self.grid.dx);
                    let ghost_coeff_y_min = Self::ghost_coefficient(bc.y_min, d, self.grid.dy);
                    let ghost_coeff_y_max = Self::ghost_coefficient(bc.y_max, d, self.grid.dy);
                    let ghost_coeff_z_min = Self::ghost_coefficient(bc.z_min, d, self.grid.dz);
                    let ghost_coeff_z_max = Self::ghost_coefficient(bc.z_max, d, self.grid.dz);

                    let d_x_minus = if i > 0 {
                        0.5 * (d + self.diffusion_coefficient[[i - 1, j, k]])
                    } else {
                        d * (1.0 - ghost_coeff_x_min)
                    };
                    let d_x_plus = if i + 1 < nx {
                        0.5 * (d + self.diffusion_coefficient[[i + 1, j, k]])
                    } else {
                        d * (1.0 - ghost_coeff_x_max)
                    };
                    let d_y_minus = if j > 0 {
                        0.5 * (d + self.diffusion_coefficient[[i, j - 1, k]])
                    } else {
                        d * (1.0 - ghost_coeff_y_min)
                    };
                    let d_y_plus = if j + 1 < ny {
                        0.5 * (d + self.diffusion_coefficient[[i, j + 1, k]])
                    } else {
                        d * (1.0 - ghost_coeff_y_max)
                    };
                    let d_z_minus = if k > 0 {
                        0.5 * (d + self.diffusion_coefficient[[i, j, k - 1]])
                    } else {
                        d * (1.0 - ghost_coeff_z_min)
                    };
                    let d_z_plus = if k + 1 < nz {
                        0.5 * (d + self.diffusion_coefficient[[i, j, k + 1]])
                    } else {
                        d * (1.0 - ghost_coeff_z_max)
                    };

                    let diagonal = (d_x_minus + d_x_plus) * dx2_inv
                        + (d_y_minus + d_y_plus) * dy2_inv
                        + (d_z_minus + d_z_plus) * dz2_inv
                        + mu_a;

                    // Preconditioner is inverse of diagonal
                    preconditioner[[i, j, k]] = if diagonal > 1e-30 {
                        1.0 / diagonal
                    } else {
                        1.0
                    };
                }
            }
        }

        preconditioner
    }

    /// Get grid reference
    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    /// Get diffusion coefficient field
    pub fn diffusion_coefficient(&self) -> &Array3<f64> {
        &self.diffusion_coefficient
    }

    /// Get absorption coefficient field
    pub fn absorption_coefficient(&self) -> &Array3<f64> {
        &self.absorption_coefficient
    }
}

/// Analytical validation solutions for testing
pub mod analytical {
    use super::*;

    /// Infinite medium Green's function solution
    ///
    /// Point source at origin in infinite homogeneous medium:
    /// ```text
    /// Φ(r) = (P₀ / (4π D r)) exp(-μ_eff r)
    /// ```
    /// where μ_eff = √(3μₐ(μₐ + μₛ'))
    ///
    /// # Reference
    /// - Contini et al. (1997): "Photon migration through a turbid slab"
    pub fn infinite_medium_point_source(
        r: f64,
        source_power: f64,
        optical_properties: OpticalPropertyData,
    ) -> f64 {
        let mu_a = optical_properties.absorption_coefficient;
        let mu_s_prime = optical_properties.reduced_scattering();
        let d = 1.0 / (3.0 * (mu_a + mu_s_prime));

        let mu_eff = (3.0 * mu_a * (mu_a + mu_s_prime)).sqrt();

        if r < 1e-10 {
            // Singularity at origin
            return f64::INFINITY;
        }

        (source_power / (4.0 * std::f64::consts::PI * d * r)) * (-mu_eff * r).exp()
    }

    /// Semi-infinite medium solution (diffuse reflectance)
    ///
    /// Extrapolated boundary source at depth z_0 = 1/μ_tr:
    /// ```text
    /// Φ(ρ, z) = (3P₀ μ_tr / (4π)) [exp(-μ_eff r₁)/r₁ - exp(-μ_eff r₂)/r₂]
    /// ```
    /// where r₁ = √(ρ² + (z - z₀)²), r₂ = √(ρ² + (z + z₀ + 4AD)²)
    pub fn semi_infinite_medium(
        rho: f64,
        z: f64,
        source_power: f64,
        optical_properties: OpticalPropertyData,
        boundary_parameter: f64,
    ) -> f64 {
        let mu_a = optical_properties.absorption_coefficient;
        let mu_s_prime = optical_properties.reduced_scattering();
        let d = 1.0 / (3.0 * (mu_a + mu_s_prime));
        let mu_tr = mu_a + mu_s_prime;

        let mu_eff = (3.0 * mu_a * mu_tr).sqrt();
        let z0 = 1.0 / mu_tr;

        let r1 = (rho * rho + (z - z0) * (z - z0)).sqrt();
        let r2 = (rho * rho + (z + z0 + 4.0 * boundary_parameter * d).powi(2)).sqrt();

        let prefactor = 3.0 * source_power * mu_tr / (4.0 * std::f64::consts::PI);

        if r1 < 1e-10 && r2 < 1e-10 {
            return 0.0;
        }

        prefactor * ((-mu_eff * r1).exp() / r1.max(1e-10) - (-mu_eff * r2).exp() / r2.max(1e-10))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytical_infinite_medium() {
        let tissue = OpticalPropertyData::soft_tissue();
        let power = 1.0; // 1 W point source

        // Test at various distances
        let distances = [0.001, 0.005, 0.01, 0.02];
        for &r in &distances {
            let fluence = analytical::infinite_medium_point_source(r, power, tissue);
            // Fluence should decay exponentially with distance
            assert!(fluence > 0.0);
            assert!(fluence.is_finite());
        }

        // Test monotonic decay
        let fluence_near = analytical::infinite_medium_point_source(0.01, power, tissue);
        let fluence_far = analytical::infinite_medium_point_source(0.02, power, tissue);
        assert!(
            fluence_near > fluence_far,
            "Fluence should decay with distance"
        );
    }

    #[test]
    fn test_solver_uniform_medium() -> Result<()> {
        // Small grid for fast testing
        let grid = Grid::new(20, 20, 20, 1e-3, 1e-3, 1e-3)?;
        let tissue = OpticalPropertyData::soft_tissue();

        let config = DiffusionSolverConfig {
            max_iterations: 1000,
            tolerance: 1e-4,
            boundary_parameter: 2.0,
            boundary_conditions: None,
            verbose: false,
        };

        let solver = DiffusionSolver::uniform(grid.clone(), tissue, config)?;

        // Point source at center
        let (nx, ny, nz) = grid.dimensions();
        let mut source = Array3::zeros((nx, ny, nz));
        source[[nx / 2, ny / 2, nz / 2]] = 1e6; // 1 MW/m³ localized source

        let fluence = solver.solve(&source)?;

        // Basic physical checks
        assert!(
            fluence.iter().all(|&x| x >= 0.0),
            "Fluence must be non-negative"
        );
        assert!(
            fluence.iter().any(|&x| x > 0.0),
            "Fluence should be non-zero"
        );

        // Maximum should be near source
        let max_idx = fluence
            .indexed_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        let dist_from_center = ((max_idx.0 as isize - nx as isize / 2).pow(2)
            + (max_idx.1 as isize - ny as isize / 2).pow(2)
            + (max_idx.2 as isize - nz as isize / 2).pow(2)) as f64;
        assert!(
            dist_from_center < 10.0,
            "Maximum fluence should be near source location"
        );

        Ok(())
    }

    #[test]
    fn test_solver_symmetry() -> Result<()> {
        // Test radial symmetry for point source
        let grid = Grid::new(30, 30, 30, 1e-3, 1e-3, 1e-3)?;
        let tissue = OpticalPropertyData::soft_tissue();

        let config = DiffusionSolverConfig {
            max_iterations: 2000,
            tolerance: 1e-5,
            boundary_parameter: 2.0,
            boundary_conditions: None,
            verbose: false,
        };

        let solver = DiffusionSolver::uniform(grid.clone(), tissue, config)?;

        let (nx, ny, nz) = grid.dimensions();
        let mut source = Array3::zeros((nx, ny, nz));
        let center = (nx / 2, ny / 2, nz / 2);
        source[[center.0, center.1, center.2]] = 1e6;

        let fluence = solver.solve(&source)?;

        // Check symmetry: points equidistant from center should have similar fluence
        let r_test = 5; // Test at radius of 5 grid points
        let test_points = [
            (center.0 + r_test, center.1, center.2),
            (center.0 - r_test, center.1, center.2),
            (center.0, center.1 + r_test, center.2),
            (center.0, center.1 - r_test, center.2),
        ];

        let fluence_values: Vec<f64> = test_points
            .iter()
            .filter(|&&(i, j, k)| i < nx && j < ny && k < nz)
            .map(|&(i, j, k)| fluence[[i, j, k]])
            .collect();

        if fluence_values.len() >= 2 {
            let mean = fluence_values.iter().sum::<f64>() / fluence_values.len() as f64;
            let max_deviation = fluence_values
                .iter()
                .map(|&f| (f - mean).abs() / mean)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);

            assert!(
                max_deviation < 0.2,
                "Symmetry check failed: max deviation = {:.1}%",
                max_deviation * 100.0
            );
        }

        Ok(())
    }

    #[test]
    fn test_heterogeneous_medium() -> Result<()> {
        // Two-layer medium: soft tissue + tumor region
        let grid = Grid::new(20, 20, 20, 1e-3, 1e-3, 1e-3)?;
        let (nx, ny, nz) = grid.dimensions();

        let tissue = OpticalPropertyData::soft_tissue();
        let tumor = OpticalPropertyData::tumor();

        let mut optical_map = Array3::from_elem((nx, ny, nz), tissue);

        // Add tumor sphere in center
        let center = (nx / 2, ny / 2, nz / 2);
        let radius = 5;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let dist_sq = (i as isize - center.0 as isize).pow(2)
                        + (j as isize - center.1 as isize).pow(2)
                        + (k as isize - center.2 as isize).pow(2);
                    if dist_sq <= (radius as isize).pow(2) {
                        optical_map[[i, j, k]] = tumor;
                    }
                }
            }
        }

        let config = DiffusionSolverConfig {
            max_iterations: 2000,
            tolerance: 1e-4,
            boundary_parameter: 2.0,
            boundary_conditions: None,
            verbose: false,
        };

        let solver = DiffusionSolver::new(grid, optical_map, config)?;

        // Point source at center (inside tumor)
        let mut source = Array3::zeros((nx, ny, nz));
        source[[center.0, center.1, center.2]] = 1e6;

        let fluence = solver.solve(&source)?;

        // Fluence should be enhanced in tumor region (higher absorption → more photoacoustic signal)
        assert!(fluence[[center.0, center.1, center.2]] > 0.0);
        assert!(fluence.iter().all(|&x| x >= 0.0));

        Ok(())
    }
}
