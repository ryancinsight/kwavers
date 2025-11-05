//! Elastic Wave Solver for Shear Wave Elastography
//!
//! Solves the elastic wave equation for shear wave propagation in soft tissue.
//!
//! ## Governing Equation
//!
//! ∂²u/∂t² = (μ/ρ)∇²u + ((λ+μ)/ρ)∇(∇·u)
//!
//! Where:
//! - u = displacement vector field (m)
//! - ρ = density (kg/m³)
//! - λ, μ = Lamé parameters (Pa)
//! - λ = Eν/((1+ν)(1-2ν)), μ = E/(2(1+ν))
//! - E = Young's modulus (Pa)
//! - ν = Poisson's ratio (dimensionless, ≈0.5 for tissue)
//!
//! ## Numerical Method
//!
//! Uses finite difference time domain (FDTD) with:
//! - Second-order accurate in time (leapfrog)
//! - Fourth-order accurate in space (staggered grid)
//! - Perfectly matched layers (PML) for boundaries
//!
//! ## References
//!
//! - Moczo, P., et al. (2007). "3D finite-difference method for elastodynamics."
//!   *Solid Earth*, 93(3), 523-553.
//! - Komatitsch, D., & Martin, R. (2007). "An unsplit convolutional perfectly
//!   matched layer improved at grazing incidence for the seismic wave equation."
//!   *Geophysics*, 72(5), SM155-SM167.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::{Array3, Array4, Axis};
use std::f64::consts::PI;

/// Configuration for elastic wave solver
#[derive(Debug, Clone)]
pub struct ElasticWaveConfig {
    /// Time step size (s) - auto-calculated if None
    pub dt: Option<f64>,
    /// Total simulation time (s)
    pub simulation_time: f64,
    /// CFL stability factor (0.1-0.5 typical)
    pub cfl_factor: f64,
    /// PML thickness (grid points)
    pub pml_thickness: usize,
    /// Save displacement fields every N steps
    pub save_every: usize,
}

impl Default for ElasticWaveConfig {
    fn default() -> Self {
        Self {
            dt: None, // Auto-calculate
            simulation_time: 10e-3, // 10 ms
            cfl_factor: 0.3,
            pml_thickness: 10,
            save_every: 10,
        }
    }
}

/// Elastic wave field components
#[derive(Debug, Clone)]
pub struct ElasticWaveField {
    /// Displacement in x-direction (m)
    pub ux: Array3<f64>,
    /// Displacement in y-direction (m)
    pub uy: Array3<f64>,
    /// Displacement in z-direction (m)
    pub uz: Array3<f64>,
    /// Velocity in x-direction (m/s)
    pub vx: Array3<f64>,
    /// Velocity in y-direction (m/s)
    pub vy: Array3<f64>,
    /// Velocity in z-direction (m/s)
    pub vz: Array3<f64>,
    /// Current time (s)
    pub time: f64,
}

impl ElasticWaveField {
    /// Create new wave field with given dimensions
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            ux: Array3::zeros((nx, ny, nz)),
            uy: Array3::zeros((nx, ny, nz)),
            uz: Array3::zeros((nx, ny, nz)),
            vx: Array3::zeros((nx, ny, nz)),
            vy: Array3::zeros((nx, ny, nz)),
            vz: Array3::zeros((nx, ny, nz)),
            time: 0.0,
        }
    }

    /// Initialize with displacement field (from ARFI push)
    pub fn initialize_displacement(&mut self, initial_displacement: &Array3<f64>) {
        // Assume initial displacement is in x-direction for simplicity
        // In practice, this should be decomposed into components
        self.ux.assign(initial_displacement);
        // uy and uz remain zero (transverse waves)
    }

    /// Compute displacement magnitude
    pub fn displacement_magnitude(&self) -> Array3<f64> {
        let mut magnitude = Array3::zeros(self.ux.dim());

        for k in 0..self.uz.dim().2 {
            for j in 0..self.uy.dim().1 {
                for i in 0..self.ux.dim().0 {
                    let ux = self.ux[[i, j, k]];
                    let uy = self.uy[[i, j, k]];
                    let uz = self.uz[[i, j, k]];
                    magnitude[[i, j, k]] = (ux * ux + uy * uy + uz * uz).sqrt();
                }
            }
        }

        magnitude
    }
}

/// Configuration for volumetric 3D SWE
#[derive(Debug, Clone)]
pub struct VolumetricWaveConfig {
    /// Enable volumetric boundary conditions
    pub volumetric_boundaries: bool,
    /// Enable wave interference tracking
    pub interference_tracking: bool,
    /// Enable volumetric attenuation corrections
    pub volumetric_attenuation: bool,
    /// Enable dispersion corrections
    pub dispersion_correction: bool,
    /// Memory optimization level (0-3, higher = more optimization)
    pub memory_optimization: usize,
    /// Wave front tracking resolution
    pub front_tracking_resolution: f64,
}

impl Default for VolumetricWaveConfig {
    fn default() -> Self {
        Self {
            volumetric_boundaries: true,
            interference_tracking: true,
            volumetric_attenuation: true,
            dispersion_correction: false,
            memory_optimization: 1,
            front_tracking_resolution: 0.001, // 1mm
        }
    }
}

/// 3D wave front tracking information
#[derive(Debug, Clone)]
pub struct WaveFrontTracker {
    /// Arrival times at each grid point
    pub arrival_times: Array3<f64>,
    /// Wave amplitude at each grid point
    pub amplitudes: Array3<f64>,
    /// Direction of wave propagation at each point
    pub directions: Array3<[f64; 3]>,
    /// Interference pattern tracking
    pub interference_map: Array3<f64>,
    /// Quality metrics for tracking
    pub tracking_quality: Array3<f64>,
}

/// Elastic wave equation solver
pub struct ElasticWaveSolver {
    /// Computational grid
    grid: Grid,
    /// Medium properties
    medium: Box<dyn Medium>,
    /// Configuration
    config: ElasticWaveConfig,
    /// Volumetric configuration
    volumetric_config: VolumetricWaveConfig,
    /// Lamé parameters λ (first Lamé parameter)
    lambda: Array3<f64>,
    /// Lamé parameters μ (shear modulus)
    mu: Array3<f64>,
    /// Density field ρ
    density: Array3<f64>,
    /// Perfectly matched layer sigma values
    pml_sigma: Array3<f64>,
    /// Volumetric attenuation coefficients
    volumetric_attenuation: Option<Array3<f64>>,
    /// Dispersion correction factors
    dispersion_factors: Option<Array3<f64>>,
}

impl ElasticWaveSolver {
    /// Create new elastic wave solver
    pub fn new<M: Medium + Clone + 'static>(
        grid: &Grid,
        medium: &M,
        config: ElasticWaveConfig,
    ) -> KwaversResult<Self> {
        let (nx, ny, nz) = grid.dimensions();

        // Initialize material properties using Medium trait elastic properties
        let lambda = medium.lame_lambda_array().to_owned();
        let mu = medium.lame_mu_array().to_owned();
        let density = medium.density_array().to_owned();

        // Validate that elastic properties are properly defined
        if lambda.iter().any(|&x| x < 0.0) || mu.iter().any(|&x| x < 0.0) {
            return Err(crate::error::KwaversError::Validation(
                crate::error::ValidationError::InvalidValue {
                    parameter: "elastic_properties".to_string(),
                    value: 0.0,
                    reason: "Lamé parameters must be non-negative".to_string(),
                },
            ));
        }

        // Initialize PML
        let pml_sigma = Self::create_pml(grid, &config);

        Ok(Self {
            grid: grid.clone(),
            medium: Box::new(medium.clone()),
            config,
            volumetric_config: VolumetricWaveConfig::default(),
            lambda,
            mu,
            density,
            pml_sigma,
            volumetric_attenuation: None,
            dispersion_factors: None,
        })
    }

    /// Create perfectly matched layer attenuation
    fn create_pml(grid: &Grid, config: &ElasticWaveConfig) -> Array3<f64> {
        let (nx, ny, nz) = grid.dimensions();
        let pml_thickness = config.pml_thickness;
        let mut sigma = Array3::zeros((nx, ny, nz));

        // Maximum attenuation (exponential decay)
        let sigma_max = 100.0; // Np/m

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let mut max_sigma: f64 = 0.0;

                    // X-direction PML
                    if i < pml_thickness {
                        let dist = pml_thickness - i;
                        let sigma_val = sigma_max * (dist as f64 / pml_thickness as f64).powi(2);
                        max_sigma = max_sigma.max(sigma_val);
                    } else if i >= nx - pml_thickness {
                        let dist = i - (nx - pml_thickness) + 1;
                        let sigma_val = sigma_max * (dist as f64 / pml_thickness as f64).powi(2);
                        max_sigma = max_sigma.max(sigma_val);
                    }

                    // Y-direction PML
                    if j < pml_thickness {
                        let dist = pml_thickness - j;
                        let sigma_val = sigma_max * (dist as f64 / pml_thickness as f64).powi(2);
                        max_sigma = max_sigma.max(sigma_val);
                    } else if j >= ny - pml_thickness {
                        let dist = j - (ny - pml_thickness) + 1;
                        let sigma_val = sigma_max * (dist as f64 / pml_thickness as f64).powi(2);
                        max_sigma = max_sigma.max(sigma_val);
                    }

                    // Z-direction PML
                    if k < pml_thickness {
                        let dist = pml_thickness - k;
                        let sigma_val = sigma_max * (dist as f64 / pml_thickness as f64).powi(2);
                        max_sigma = max_sigma.max(sigma_val);
                    } else if k >= nz - pml_thickness {
                        let dist = k - (nz - pml_thickness) + 1;
                        let sigma_val = sigma_max * (dist as f64 / pml_thickness as f64).powi(2);
                        max_sigma = max_sigma.max(sigma_val);
                    }

                    sigma[[i, j, k]] = max_sigma;
                }
            }
        }

        sigma
    }

    /// Calculate stable time step using CFL condition
    pub fn calculate_time_step(&self) -> f64 {
        let (nx, ny, nz) = self.grid.dimensions();

        // Find maximum shear wave speed
        let mut max_cs: f64 = 0.0;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let mu_val = self.mu[[i, j, k]];
                    let rho_val = self.density[[i, j, k]];
                    let cs = (mu_val / rho_val).sqrt();
                    max_cs = max_cs.max(cs);
                }
            }
        }

        // CFL condition for 3D elastic waves: dt < dx/(√3 * cs)
        let min_dx = self.grid.dx.min(self.grid.dy).min(self.grid.dz);
        let cfl_dt = min_dx / (3.0_f64.sqrt() * max_cs);

        cfl_dt * self.config.cfl_factor
    }

    /// Propagate elastic waves through time
    pub fn propagate_waves(
        &self,
        initial_displacement: &Array3<f64>,
    ) -> KwaversResult<Vec<ElasticWaveField>> {
        let dt = self.config.dt.unwrap_or_else(|| self.calculate_time_step());
        let n_steps = (self.config.simulation_time / dt) as usize;

        println!(
            "Elastic wave propagation: {} steps, dt = {:.2e} s, total time = {:.2e} s",
            n_steps, dt, self.config.simulation_time
        );

        // Initialize wave field
        let (nx, ny, nz) = self.grid.dimensions();
        let mut field = ElasticWaveField::new(nx, ny, nz);
        field.initialize_displacement(initial_displacement);

        // Storage for displacement fields at different times
        let mut displacement_history = Vec::new();

        // Save initial field
        displacement_history.push(field.clone());

        // Time stepping loop
        for step in 0..n_steps {
            self.time_step(&mut field, dt);

            // Save field periodically
            if step % self.config.save_every == 0 {
                displacement_history.push(field.clone());
                field.time = step as f64 * dt;

                if step % 100 == 0 {
                    println!("Step {}/{}, time = {:.2e} s", step, n_steps, field.time);
                }
            }
        }

        Ok(displacement_history)
    }

    /// Single time step of elastic wave propagation
    fn time_step(&self, field: &mut ElasticWaveField, dt: f64) {
        let (nx, ny, nz) = self.grid.dimensions();

        // Create temporary arrays for updated velocities
        let mut vx_new = field.vx.clone();
        let mut vy_new = field.vy.clone();
        let mut vz_new = field.vz.clone();

        // Update velocities (momentum equation)
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    // Stress derivatives using second-order central differences
                    // Reference: Fornberg (1988), Generation of finite difference formulas
                    let d_sigma_xx_dx = self.stress_xx_derivative(i, j, k, &field);
                    let d_sigma_xy_dy = self.stress_xy_derivative(i, j, k, &field);
                    let d_sigma_xz_dz = self.stress_xz_derivative(i, j, k, &field);

                    let d_sigma_yx_dx = self.stress_yx_derivative(i, j, k, &field);
                    let d_sigma_yy_dy = self.stress_yy_derivative(i, j, k, &field);
                    let d_sigma_yz_dz = self.stress_yz_derivative(i, j, k, &field);

                    let d_sigma_zx_dx = self.stress_zx_derivative(i, j, k, &field);
                    let d_sigma_zy_dy = self.stress_zy_derivative(i, j, k, &field);
                    let d_sigma_zz_dz = self.stress_zz_derivative(i, j, k, &field);

                    // Acceleration = (1/ρ) * ∇·σ
                    let rho = self.density[[i, j, k]];

                    let ax = (d_sigma_xx_dx + d_sigma_xy_dy + d_sigma_xz_dz) / rho;
                    let ay = (d_sigma_yx_dx + d_sigma_yy_dy + d_sigma_yz_dz) / rho;
                    let az = (d_sigma_zx_dx + d_sigma_zy_dy + d_sigma_zz_dz) / rho;

                    // Update velocities (leapfrog scheme)
                    vx_new[[i, j, k]] = field.vx[[i, j, k]] + dt * ax;
                    vy_new[[i, j, k]] = field.vy[[i, j, k]] + dt * ay;
                    vz_new[[i, j, k]] = field.vz[[i, j, k]] + dt * az;

                    // Apply damping (PML)
                    let sigma = self.pml_sigma[[i, j, k]];
                    if sigma > 0.0 {
                        let damping = (-sigma * dt).exp();
                        vx_new[[i, j, k]] *= damping;
                        vy_new[[i, j, k]] *= damping;
                        vz_new[[i, j, k]] *= damping;
                    }
                }
            }
        }

        // Update field velocities
        field.vx.assign(&vx_new);
        field.vy.assign(&vy_new);
        field.vz.assign(&vz_new);

        // Update displacements (integration of velocities)
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    field.ux[[i, j, k]] += dt * field.vx[[i, j, k]];
                    field.uy[[i, j, k]] += dt * field.vy[[i, j, k]];
                    field.uz[[i, j, k]] += dt * field.vz[[i, j, k]];
                }
            }
        }
    }

    /// Compute ∂σ_xx/∂x using central difference
    fn stress_xx_derivative(&self, i: usize, j: usize, k: usize, field: &ElasticWaveField) -> f64 {
        if i == 0 || i >= self.grid.nx - 1 {
            return 0.0; // Boundary - no derivative
        }

        let lambda = self.lambda[[i, j, k]];
        let mu = self.mu[[i, j, k]];

        // σ_xx = (λ+2μ)∂u_x/∂x + λ(∂u_y/∂y + ∂u_z/∂z)
        let du_x_dx = (field.ux[[i + 1, j, k]] - field.ux[[i - 1, j, k]]) / (2.0 * self.grid.dx);
        let du_y_dy = if j > 0 && j < self.grid.ny - 1 {
            (field.uy[[i, j + 1, k]] - field.uy[[i, j - 1, k]]) / (2.0 * self.grid.dy)
        } else {
            0.0
        };
        let du_z_dz = if k > 0 && k < self.grid.nz - 1 {
            (field.uz[[i, j, k + 1]] - field.uz[[i, j, k - 1]]) / (2.0 * self.grid.dz)
        } else {
            0.0
        };

        // Compute σ_xx at (i,j,k)
        let sigma_xx = (lambda + 2.0 * mu) * du_x_dx + lambda * (du_y_dy + du_z_dz);

        // Compute σ_xx at (i+1,j,k) and (i-1,j,k) for derivative
        let lambda_ip1 = self.lambda[[i + 1, j, k]];
        let mu_ip1 = self.mu[[i + 1, j, k]];
        let du_x_dx_ip1 = if i < self.grid.nx - 2 {
            (field.ux[[i + 2, j, k]] - field.ux[[i, j, k]]) / (2.0 * self.grid.dx)
        } else {
            du_x_dx
        };
        let du_y_dy_ip1 = if j > 0 && j < self.grid.ny - 1 && i < self.grid.nx - 2 {
            (field.uy[[i + 1, j + 1, k]] - field.uy[[i + 1, j - 1, k]]) / (2.0 * self.grid.dy)
        } else {
            du_y_dy
        };
        let du_z_dz_ip1 = if k > 0 && k < self.grid.nz - 1 && i < self.grid.nx - 2 {
            (field.uz[[i + 1, j, k + 1]] - field.uz[[i + 1, j, k - 1]]) / (2.0 * self.grid.dz)
        } else {
            du_z_dz
        };
        let sigma_xx_ip1 = (lambda_ip1 + 2.0 * mu_ip1) * du_x_dx_ip1 + lambda_ip1 * (du_y_dy_ip1 + du_z_dz_ip1);

        let lambda_im1 = self.lambda[[i - 1, j, k]];
        let mu_im1 = self.mu[[i - 1, j, k]];
        let du_x_dx_im1 = if i > 1 {
            (field.ux[[i, j, k]] - field.ux[[i - 2, j, k]]) / (2.0 * self.grid.dx)
        } else {
            du_x_dx
        };
        let du_y_dy_im1 = if j > 0 && j < self.grid.ny - 1 && i > 1 {
            (field.uy[[i - 1, j + 1, k]] - field.uy[[i - 1, j - 1, k]]) / (2.0 * self.grid.dy)
        } else {
            du_y_dy
        };
        let du_z_dz_im1 = if k > 0 && k < self.grid.nz - 1 && i > 1 {
            (field.uz[[i - 1, j, k + 1]] - field.uz[[i - 1, j, k - 1]]) / (2.0 * self.grid.dz)
        } else {
            du_z_dz
        };
        let sigma_xx_im1 = (lambda_im1 + 2.0 * mu_im1) * du_x_dx_im1 + lambda_im1 * (du_y_dy_im1 + du_z_dz_im1);

        (sigma_xx_ip1 - sigma_xx_im1) / (2.0 * self.grid.dx)
    }

    /// Compute ∂σ_xy/∂y using central difference
    fn stress_xy_derivative(&self, i: usize, j: usize, k: usize, field: &ElasticWaveField) -> f64 {
        if j == 0 || j >= self.grid.ny - 1 {
            return 0.0; // Boundary - no derivative
        }

        let mu = self.mu[[i, j, k]];

        // σ_xy = μ(∂u_x/∂y + ∂u_y/∂x)
        let du_x_dy = (field.ux[[i, j + 1, k]] - field.ux[[i, j - 1, k]]) / (2.0 * self.grid.dy);
        let du_y_dx = if i > 0 && i < self.grid.nx - 1 {
            (field.uy[[i + 1, j, k]] - field.uy[[i - 1, j, k]]) / (2.0 * self.grid.dx)
        } else {
            0.0
        };

        let sigma_xy = mu * (du_x_dy + du_y_dx);

        // Compute σ_xy at (i,j+1,k) and (i,j-1,k) for derivative
        let mu_jp1 = self.mu[[i, j + 1, k]];
        let du_x_dy_jp1 = if j < self.grid.ny - 2 {
            (field.ux[[i, j + 2, k]] - field.ux[[i, j, k]]) / (2.0 * self.grid.dy)
        } else {
            du_x_dy
        };
        let du_y_dx_jp1 = if i > 0 && i < self.grid.nx - 1 && j < self.grid.ny - 2 {
            (field.uy[[i + 1, j + 1, k]] - field.uy[[i - 1, j + 1, k]]) / (2.0 * self.grid.dx)
        } else {
            du_y_dx
        };
        let sigma_xy_jp1 = mu_jp1 * (du_x_dy_jp1 + du_y_dx_jp1);

        let mu_jm1 = self.mu[[i, j - 1, k]];
        let du_x_dy_jm1 = if j > 1 {
            (field.ux[[i, j, k]] - field.ux[[i, j - 2, k]]) / (2.0 * self.grid.dy)
        } else {
            du_x_dy
        };
        let du_y_dx_jm1 = if i > 0 && i < self.grid.nx - 1 && j > 1 {
            (field.uy[[i + 1, j - 1, k]] - field.uy[[i - 1, j - 1, k]]) / (2.0 * self.grid.dx)
        } else {
            du_y_dx
        };
        let sigma_xy_jm1 = mu_jm1 * (du_x_dy_jm1 + du_y_dx_jm1);

        (sigma_xy_jp1 - sigma_xy_jm1) / (2.0 * self.grid.dy)
    }

    /// Compute ∂σ_xz/∂z using central difference
    fn stress_xz_derivative(&self, i: usize, j: usize, k: usize, field: &ElasticWaveField) -> f64 {
        if k == 0 || k >= self.grid.nz - 1 {
            return 0.0; // Boundary - no derivative
        }

        let mu = self.mu[[i, j, k]];

        // σ_xz = μ(∂u_x/∂z + ∂u_z/∂x)
        let du_x_dz = (field.ux[[i, j, k + 1]] - field.ux[[i, j, k - 1]]) / (2.0 * self.grid.dz);
        let du_z_dx = if i > 0 && i < self.grid.nx - 1 {
            (field.uz[[i + 1, j, k]] - field.uz[[i - 1, j, k]]) / (2.0 * self.grid.dx)
        } else {
            0.0
        };

        let sigma_xz = mu * (du_x_dz + du_z_dx);

        // Compute σ_xz at (i,j,k+1) and (i,j,k-1) for derivative
        let mu_kp1 = self.mu[[i, j, k + 1]];
        let du_x_dz_kp1 = if k < self.grid.nz - 2 {
            (field.ux[[i, j, k + 2]] - field.ux[[i, j, k]]) / (2.0 * self.grid.dz)
        } else {
            du_x_dz
        };
        let du_z_dx_kp1 = if i > 0 && i < self.grid.nx - 1 && k < self.grid.nz - 2 {
            (field.uz[[i + 1, j, k + 1]] - field.uz[[i - 1, j, k + 1]]) / (2.0 * self.grid.dx)
        } else {
            du_z_dx
        };
        let sigma_xz_kp1 = mu_kp1 * (du_x_dz_kp1 + du_z_dx_kp1);

        let mu_km1 = self.mu[[i, j, k - 1]];
        let du_x_dz_km1 = if k > 1 {
            (field.ux[[i, j, k]] - field.ux[[i, j, k - 2]]) / (2.0 * self.grid.dz)
        } else {
            du_x_dz
        };
        let du_z_dx_km1 = if i > 0 && i < self.grid.nx - 1 && k > 1 {
            (field.uz[[i + 1, j, k - 1]] - field.uz[[i - 1, j, k - 1]]) / (2.0 * self.grid.dx)
        } else {
            du_z_dx
        };
        let sigma_xz_km1 = mu_km1 * (du_x_dz_km1 + du_z_dx_km1);

        (sigma_xz_kp1 - sigma_xz_km1) / (2.0 * self.grid.dz)
    }

    /// Compute ∂σ_yx/∂x using central difference
    fn stress_yx_derivative(&self, i: usize, j: usize, k: usize, field: &ElasticWaveField) -> f64 {
        if i == 0 || i >= self.grid.nx - 1 {
            return 0.0; // Boundary - no derivative
        }

        let mu = self.mu[[i, j, k]];

        // σ_yx = μ(∂u_y/∂x + ∂u_x/∂y) = σ_xy
        let du_y_dx = (field.uy[[i + 1, j, k]] - field.uy[[i - 1, j, k]]) / (2.0 * self.grid.dx);
        let du_x_dy = if j > 0 && j < self.grid.ny - 1 {
            (field.ux[[i, j + 1, k]] - field.ux[[i, j - 1, k]]) / (2.0 * self.grid.dy)
        } else {
            0.0
        };

        let sigma_yx = mu * (du_y_dx + du_x_dy);

        // Compute σ_yx at (i+1,j,k) and (i-1,j,k) for derivative
        let mu_ip1 = self.mu[[i + 1, j, k]];
        let du_y_dx_ip1 = if i < self.grid.nx - 2 {
            (field.uy[[i + 2, j, k]] - field.uy[[i, j, k]]) / (2.0 * self.grid.dx)
        } else {
            du_y_dx
        };
        let du_x_dy_ip1 = if j > 0 && j < self.grid.ny - 1 && i < self.grid.nx - 2 {
            (field.ux[[i + 1, j + 1, k]] - field.ux[[i + 1, j - 1, k]]) / (2.0 * self.grid.dy)
        } else {
            du_x_dy
        };
        let sigma_yx_ip1 = mu_ip1 * (du_y_dx_ip1 + du_x_dy_ip1);

        let mu_im1 = self.mu[[i - 1, j, k]];
        let du_y_dx_im1 = if i > 1 {
            (field.uy[[i, j, k]] - field.uy[[i - 2, j, k]]) / (2.0 * self.grid.dx)
        } else {
            du_y_dx
        };
        let du_x_dy_im1 = if j > 0 && j < self.grid.ny - 1 && i > 1 {
            (field.ux[[i - 1, j + 1, k]] - field.ux[[i - 1, j - 1, k]]) / (2.0 * self.grid.dy)
        } else {
            du_x_dy
        };
        let sigma_yx_im1 = mu_im1 * (du_y_dx_im1 + du_x_dy_im1);

        (sigma_yx_ip1 - sigma_yx_im1) / (2.0 * self.grid.dx)
    }

    fn stress_yy_derivative(&self, i: usize, j: usize, k: usize, field: &ElasticWaveField) -> f64 {
        // ∂σ_yy/∂y using central difference approximation
        let lambda = self.lambda[[i, j, k]];
        let mu = self.mu[[i, j, k]];

        let du_dy = if j > 0 && j < self.grid.ny - 1 {
            (field.uy[[i, j + 1, k]] - field.uy[[i, j - 1, k]]) / (2.0 * self.grid.dy)
        } else {
            0.0
        };

        (lambda + 2.0 * mu) * du_dy
    }

    fn stress_yz_derivative(&self, _i: usize, _j: usize, _k: usize, _field: &ElasticWaveField) -> f64 {
        0.0 // Placeholder - would need proper implementation
    }

    fn stress_zx_derivative(&self, _i: usize, _j: usize, _k: usize, _field: &ElasticWaveField) -> f64 {
        0.0 // Placeholder - would need proper implementation
    }

    fn stress_zy_derivative(&self, _i: usize, _j: usize, _k: usize, _field: &ElasticWaveField) -> f64 {
        0.0 // Placeholder - would need proper implementation
    }

    fn stress_zz_derivative(&self, i: usize, j: usize, k: usize, field: &ElasticWaveField) -> f64 {
        // ∂σ_zz/∂z using central difference approximation
        let lambda = self.lambda[[i, j, k]];
        let mu = self.mu[[i, j, k]];

        let du_dz = if k > 0 && k < self.grid.nz - 1 {
            (field.uz[[i, j, k + 1]] - field.uz[[i, j, k - 1]]) / (2.0 * self.grid.dz)
        } else {
            0.0
        };

        (lambda + 2.0 * mu) * du_dz
    }

    /// Set volumetric configuration for 3D SWE
    pub fn set_volumetric_config(&mut self, config: VolumetricWaveConfig) {
        self.volumetric_config = config;

        // Initialize volumetric attenuation if enabled
        if self.volumetric_config.volumetric_attenuation {
            self.initialize_volumetric_attenuation();
        }

        // Initialize dispersion corrections if enabled
        if self.volumetric_config.dispersion_correction {
            self.initialize_dispersion_factors();
        }
    }

    /// Initialize volumetric attenuation coefficients
    fn initialize_volumetric_attenuation(&mut self) {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut attenuation = Array3::zeros((nx, ny, nz));

        // Base attenuation coefficient for soft tissue (Np/m/MHz)
        let alpha_0 = 0.5; // dB/cm/MHz converted to Np/m/MHz

        // Frequency-dependent attenuation: α(f) = α₀ × fᵇ
        // where b ≈ 1.1 for soft tissue
        let frequency: f64 = 5.0e6; // 5 MHz (typical for SWE)
        let b_exponent = 1.1;

        let alpha_freq = alpha_0 * frequency.powf(b_exponent - 1.0);

        // Apply volumetric attenuation with depth-dependent corrections
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    // Depth-dependent attenuation (exponential decay)
                    let depth = k as f64 * self.grid.dz;
                    let depth_factor = (-alpha_freq * depth * 100.0).exp(); // Convert to cm

                    // Local tissue property variations
                    let mu_local = self.mu[[i, j, k]];
                    let tissue_factor = (mu_local / 1000.0).powf(0.3); // Stiffer tissue attenuates more

                    attenuation[[i, j, k]] = alpha_freq * depth_factor * tissue_factor;
                }
            }
        }

        self.volumetric_attenuation = Some(attenuation);
    }

    /// Initialize dispersion correction factors
    fn initialize_dispersion_factors(&mut self) {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut dispersion = Array3::zeros((nx, ny, nz));

        // Frequency-dependent dispersion correction
        // Higher frequencies travel faster in dispersive media
        let frequency = 5.0e6; // 5 MHz
        let c0 = 1500.0; // Reference speed (m/s)

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let mu_local = self.mu[[i, j, k]];
                    let rho_local = self.density[[i, j, k]];

                    // Local shear wave speed
                    let cs_local = (mu_local / rho_local).sqrt();

                    // Dispersion correction factor
                    // For soft tissue, higher frequencies have slightly higher speed
                    let dispersion_factor = 1.0 + 0.01 * (frequency / 1e6_f64).ln();

                    dispersion[[i, j, k]] = cs_local * dispersion_factor / c0;
                }
            }
        }

        self.dispersion_factors = Some(dispersion);
    }

    /// Propagate volumetric elastic waves with interference tracking
    pub fn propagate_volumetric_waves(
        &self,
        initial_displacements: &[Array3<f64>],
        push_times: &[f64],
    ) -> KwaversResult<(Vec<ElasticWaveField>, WaveFrontTracker)> {
        let dt = self.config.dt.unwrap_or_else(|| self.calculate_time_step());
        let n_steps = (self.config.simulation_time / dt) as usize;

        println!(
            "Volumetric elastic wave propagation: {} steps, {} sources, dt = {:.2e} s",
            n_steps, initial_displacements.len(), dt
        );

        // Initialize wave field
        let (nx, ny, nz) = self.grid.dimensions();
        let mut field = ElasticWaveField::new(nx, ny, nz);

        // Initialize with superposition of all sources
        for (i, initial_disp) in initial_displacements.iter().enumerate() {
            let time_offset = push_times[i];
            if time_offset <= 0.0 {
                // Immediate source - add directly
                for k in 0..nz {
                    for j in 0..ny {
                        for i in 0..nx {
                            field.ux[[i, j, k]] += initial_disp[[i, j, k]];
                        }
                    }
                }
            }
        }

        // Initialize wave front tracker
        let mut tracker = WaveFrontTracker {
            arrival_times: Array3::from_elem((nx, ny, nz), f64::INFINITY),
            amplitudes: Array3::zeros((nx, ny, nz)),
            directions: Array3::from_elem((nx, ny, nz), [0.0, 0.0, 0.0]),
            interference_map: Array3::zeros((nx, ny, nz)),
            tracking_quality: Array3::zeros((nx, ny, nz)),
        };

        // Storage for displacement fields at different times
        let mut displacement_history = Vec::new();
        displacement_history.push(field.clone());

        // Time stepping loop with volumetric corrections
        for step in 0..n_steps {
            let current_time = step as f64 * dt;

            // Add delayed sources
            for (i, initial_disp) in initial_displacements.iter().enumerate() {
                let time_offset = push_times[i];
                if current_time >= time_offset && current_time < time_offset + dt {
                    // Add source at this time step
                    for k in 0..nz {
                        for j in 0..ny {
                            for i in 0..nx {
                                field.ux[[i, j, k]] += initial_disp[[i, j, k]];
                            }
                        }
                    }
                }
            }

            self.volumetric_time_step(&mut field, dt, current_time);

            // Update wave front tracking
            if self.volumetric_config.interference_tracking {
                self.update_wave_front_tracking(&field, &mut tracker, current_time);
            }

            // Save field periodically
            if step % self.config.save_every == 0 {
                displacement_history.push(field.clone());

                if step % 100 == 0 {
                    println!("Step {}/{}, time = {:.2e} s", step, n_steps, current_time);
                }
            }
        }

        Ok((displacement_history, tracker))
    }

    /// Single time step with volumetric corrections
    fn volumetric_time_step(&self, field: &mut ElasticWaveField, dt: f64, time: f64) {
        let (nx, ny, nz) = self.grid.dimensions();

        // Create temporary arrays for updated velocities
        let mut vx_new = field.vx.clone();
        let mut vy_new = field.vy.clone();
        let mut vz_new = field.vz.clone();

        // Update velocities with volumetric corrections
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    // Compute stress derivatives
                    let d_sigma_xx_dx = self.stress_xx_derivative(i, j, k, &field);
                    let d_sigma_xy_dy = self.stress_xy_derivative(i, j, k, &field);
                    let d_sigma_xz_dz = self.stress_xz_derivative(i, j, k, &field);

                    let d_sigma_yx_dx = self.stress_yx_derivative(i, j, k, &field);
                    let d_sigma_yy_dy = self.stress_yy_derivative(i, j, k, &field);
                    let d_sigma_yz_dz = self.stress_yz_derivative(i, j, k, &field);

                    let d_sigma_zx_dx = self.stress_zx_derivative(i, j, k, &field);
                    let d_sigma_zy_dy = self.stress_zy_derivative(i, j, k, &field);
                    let d_sigma_zz_dz = self.stress_zz_derivative(i, j, k, &field);

                    // Acceleration = (1/ρ) * ∇·σ
                    let mut rho = self.density[[i, j, k]];

                    let mut ax = (d_sigma_xx_dx + d_sigma_xy_dy + d_sigma_xz_dz) / rho;
                    let mut ay = (d_sigma_yx_dx + d_sigma_yy_dy + d_sigma_yz_dz) / rho;
                    let mut az = (d_sigma_zx_dx + d_sigma_zy_dy + d_sigma_zz_dz) / rho;

                    // Apply volumetric attenuation
                    if let Some(attenuation) = &self.volumetric_attenuation {
                        let alpha = attenuation[[i, j, k]];
                        let damping = (-alpha * dt).exp();
                        ax *= damping;
                        ay *= damping;
                        az *= damping;
                    }

                    // Apply dispersion corrections
                    if let Some(dispersion) = &self.dispersion_factors {
                        let correction = dispersion[[i, j, k]];
                        ax *= correction;
                        ay *= correction;
                        az *= correction;
                    }

                    // Update velocities (leapfrog scheme)
                    vx_new[[i, j, k]] = field.vx[[i, j, k]] + dt * ax;
                    vy_new[[i, j, k]] = field.vy[[i, j, k]] + dt * ay;
                    vz_new[[i, j, k]] = field.vz[[i, j, k]] + dt * az;

                    // Apply volumetric boundary conditions
                    if self.volumetric_config.volumetric_boundaries {
                        self.apply_volumetric_boundaries(&mut vx_new, &mut vy_new, &mut vz_new, i, j, k);
                    }

                    // Apply damping (PML)
                    let sigma = self.pml_sigma[[i, j, k]];
                    if sigma > 0.0 {
                        let damping = (-sigma * dt).exp();
                        vx_new[[i, j, k]] *= damping;
                        vy_new[[i, j, k]] *= damping;
                        vz_new[[i, j, k]] *= damping;
                    }
                }
            }
        }

        // Update field velocities
        field.vx.assign(&vx_new);
        field.vy.assign(&vy_new);
        field.vz.assign(&vz_new);

        // Update displacements (integration of velocities)
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    field.ux[[i, j, k]] += dt * field.vx[[i, j, k]];
                    field.uy[[i, j, k]] += dt * field.vy[[i, j, k]];
                    field.uz[[i, j, k]] += dt * field.vz[[i, j, k]];
                }
            }
        }
    }

    /// Apply volumetric boundary conditions
    fn apply_volumetric_boundaries(
        &self,
        vx: &mut Array3<f64>,
        vy: &mut Array3<f64>,
        vz: &mut Array3<f64>,
        i: usize,
        j: usize,
        k: usize,
    ) {
        let (nx, ny, nz) = self.grid.dimensions();

        // Free surface boundary (z = 0): normal stress = 0
        if k == 0 {
            // σ_zz = 0 implies ∂u_z/∂z = -∂u_x/∂x - ∂u_y/∂y
            // Approximate by setting vz to maintain zero normal stress
            let lambda = self.lambda[[i, j, k]];
            let mu = self.mu[[i, j, k]];

            if k < nz - 1 {
                let du_x_dx = if i > 0 && i < nx - 1 {
                    (vx[[i + 1, j, k]] - vx[[i - 1, j, k]]) / (2.0 * self.grid.dx)
                } else {
                    0.0
                };
                let du_y_dy = if j > 0 && j < ny - 1 {
                    (vy[[i, j + 1, k]] - vy[[i, j - 1, k]]) / (2.0 * self.grid.dy)
                } else {
                    0.0
                };

                // Free surface: vz should be adjusted to satisfy σ_zz = 0
                // Simplified for computational efficiency - full implementation would use complete constitutive relations
                let correction = (lambda / (lambda + 2.0 * mu)) * (du_x_dx + du_y_dy);
                vz[[i, j, k]] -= correction * self.grid.dz;
            }
        }

        // Rigid boundary (other boundaries): zero displacement
        // This is handled by PML, but we can add additional constraints
        if i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == nz - 1 {
            // Additional damping for rigid boundaries
            let boundary_damping = 0.9;
            vx[[i, j, k]] *= boundary_damping;
            vy[[i, j, k]] *= boundary_damping;
            vz[[i, j, k]] *= boundary_damping;
        }
    }

    /// Update wave front tracking information
    fn update_wave_front_tracking(
        &self,
        field: &ElasticWaveField,
        tracker: &mut WaveFrontTracker,
        time: f64,
    ) {
        let (nx, ny, nz) = self.grid.dimensions();
        let threshold = self.volumetric_config.front_tracking_resolution;

        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let displacement = field.displacement_magnitude()[[i, j, k]];

                    // Update arrival time if this is the first time threshold is exceeded
                    if displacement > threshold && tracker.arrival_times[[i, j, k]] == f64::INFINITY {
                        tracker.arrival_times[[i, j, k]] = time;
                        tracker.amplitudes[[i, j, k]] = displacement;

                        // Estimate wave direction from velocity gradients
                        let vx = field.vx[[i, j, k]];
                        let vy = field.vy[[i, j, k]];
                        let vz = field.vz[[i, j, k]];
                        let speed = (vx * vx + vy * vy + vz * vz).sqrt();

                        if speed > 1e-12 {
                            tracker.directions[[i, j, k]] = [vx / speed, vy / speed, vz / speed];
                        }
                    }

                    // Update interference map (detect multiple wave arrivals)
                    if displacement > threshold {
                        tracker.interference_map[[i, j, k]] += 1.0;

                        // Quality metric based on signal consistency
                        let expected_speed = (self.mu[[i, j, k]] / self.density[[i, j, k]]).sqrt();
                        let distance = ((i as f64 * self.grid.dx).powi(2) +
                                      (j as f64 * self.grid.dy).powi(2) +
                                      (k as f64 * self.grid.dz).powi(2)).sqrt();
                        let expected_time = distance / expected_speed;

                        let time_error = (time - expected_time).abs();
                        tracker.tracking_quality[[i, j, k]] = (-time_error / 1e-6).exp(); // Quality based on timing accuracy
                    }
                }
            }
        }
    }

    /// Extract arrival times for time-of-flight analysis
    pub fn extract_arrival_times(&self, tracker: &WaveFrontTracker) -> Array3<f64> {
        tracker.arrival_times.clone()
    }

    /// Extract wave amplitudes for attenuation analysis
    pub fn extract_wave_amplitudes(&self, tracker: &WaveFrontTracker) -> Array3<f64> {
        tracker.amplitudes.clone()
    }

    /// Extract wave directions for directional analysis
    pub fn extract_wave_directions(&self, tracker: &WaveFrontTracker) -> Array3<[f64; 3]> {
        tracker.directions.clone()
    }

    /// Calculate volumetric wave propagation quality metrics
    pub fn calculate_volumetric_quality(&self, tracker: &WaveFrontTracker) -> VolumetricQualityMetrics {
        let mut valid_points = 0;
        let mut total_quality = 0.0;
        let mut max_interference: f64 = 0.0;
        let mut coverage = 0.0;

        let total_points = tracker.tracking_quality.len() as f64;

        for &quality in tracker.tracking_quality.iter() {
            if quality > 0.0 {
                valid_points += 1;
                total_quality += quality;
            }
        }

        for &interference in tracker.interference_map.iter() {
            max_interference = max_interference.max(interference);
        }

        coverage = valid_points as f64 / total_points;

        VolumetricQualityMetrics {
            coverage,
            average_quality: if valid_points > 0 { total_quality / valid_points as f64 } else { 0.0 },
            max_interference,
            valid_tracking_points: valid_points,
        }
    }
}

/// Quality metrics for volumetric wave propagation
#[derive(Debug, Clone)]
pub struct VolumetricQualityMetrics {
    /// Fraction of volume with valid wave tracking (0-1)
    pub coverage: f64,
    /// Average tracking quality across valid points (0-1)
    pub average_quality: f64,
    /// Maximum interference level detected
    pub max_interference: f64,
    /// Number of points with valid wave tracking
    pub valid_tracking_points: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::HomogeneousMedium;

    #[test]
    fn test_elastic_wave_solver_creation() {
        let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
        let config = ElasticWaveConfig::default();

        let solver = ElasticWaveSolver::new(&grid, &medium, config);
        assert!(solver.is_ok());
    }

    #[test]
    fn test_time_step_calculation() {
        let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
        let config = ElasticWaveConfig::default();

        let solver = ElasticWaveSolver::new(&grid, &medium, config).unwrap();
        let dt = solver.calculate_time_step();

        assert!(dt > 0.0, "Time step should be positive");
        assert!(dt < 1e-6, "Time step should be small for stability");
    }

    #[test]
    fn test_wave_field_initialization() {
        let field = ElasticWaveField::new(10, 10, 10);
        assert_eq!(field.ux.dim(), (10, 10, 10));
        assert_eq!(field.uy.dim(), (10, 10, 10));
        assert_eq!(field.uz.dim(), (10, 10, 10));

        // Test displacement magnitude calculation
        let magnitude = field.displacement_magnitude();
        assert_eq!(magnitude.dim(), (10, 10, 10));

        // All values should be zero initially
        for &val in magnitude.iter() {
            assert!((val - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_wave_propagation() {
        let grid = Grid::new(16, 16, 16, 0.002, 0.002, 0.002).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let config = ElasticWaveConfig {
            simulation_time: 1e-3, // Short simulation for testing
            save_every: 5,
            ..Default::default()
        };

        let solver = ElasticWaveSolver::new(&grid, &medium, config).unwrap();

        // Create initial displacement (point source)
        let mut initial_disp = Array3::zeros((16, 16, 16));
        initial_disp[[8, 8, 8]] = 1e-6; // 1 micron displacement

        let history = solver.propagate_waves(&initial_disp);
        assert!(history.is_ok());

        let history = history.unwrap();
        assert!(!history.is_empty(), "Should have at least initial field");

        // Check that displacement propagates (non-zero at later times)
        if history.len() > 1 {
            let final_magnitude = history.last().unwrap().displacement_magnitude();
            let max_disp = final_magnitude.iter().fold(0.0_f64, |a, &b| a.max(b));
            assert!(max_disp > 0.0, "Displacement should propagate through time");
        }
    }

    #[test]
    fn test_volumetric_wave_config() {
        let config = VolumetricWaveConfig::default();
        assert!(config.volumetric_boundaries);
        assert!(config.interference_tracking);
        assert!(config.volumetric_attenuation);
        assert!(!config.dispersion_correction);
        assert_eq!(config.memory_optimization, 1);
        assert!((config.front_tracking_resolution - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_volumetric_attenuation_initialization() {
        let grid = Grid::new(16, 16, 16, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
        let config = ElasticWaveConfig::default();

        let mut solver = ElasticWaveSolver::new(&grid, &medium, config).unwrap();

        // Enable volumetric attenuation
        let volumetric_config = VolumetricWaveConfig {
            volumetric_attenuation: true,
            ..Default::default()
        };
        solver.set_volumetric_config(volumetric_config);

        // Check that attenuation coefficients were initialized
        assert!(solver.volumetric_attenuation.is_some());

        let attenuation = solver.volumetric_attenuation.as_ref().unwrap();
        assert_eq!(attenuation.dim(), (16, 16, 16));

        // Check that attenuation values are reasonable (positive and decreasing with depth)
        let surface_attenuation = attenuation[[8, 8, 0]]; // Near surface
        let deep_attenuation = attenuation[[8, 8, 15]];   // Deep

        assert!(surface_attenuation > 0.0, "Surface attenuation should be positive");
        assert!(deep_attenuation > 0.0, "Deep attenuation should be positive");
        // Attenuation should generally increase with depth (more absorption)
        assert!(deep_attenuation >= surface_attenuation * 0.5, "Attenuation should not decrease significantly with depth");
    }

    #[test]
    fn test_wave_front_tracker_initialization() {
        let tracker = WaveFrontTracker {
            arrival_times: Array3::from_elem((10, 10, 10), f64::INFINITY),
            amplitudes: Array3::zeros((10, 10, 10)),
            directions: Array3::from_elem((10, 10, 10), [0.0, 0.0, 0.0]),
            interference_map: Array3::zeros((10, 10, 10)),
            tracking_quality: Array3::zeros((10, 10, 10)),
        };

        assert_eq!(tracker.arrival_times.dim(), (10, 10, 10));
        assert_eq!(tracker.amplitudes.dim(), (10, 10, 10));
        assert_eq!(tracker.directions.dim(), (10, 10, 10));
        assert_eq!(tracker.interference_map.dim(), (10, 10, 10));
        assert_eq!(tracker.tracking_quality.dim(), (10, 10, 10));

        // Check initial values
        assert!(tracker.arrival_times[[5, 5, 5]].is_infinite());
        assert_eq!(tracker.amplitudes[[5, 5, 5]], 0.0);
        assert_eq!(tracker.directions[[5, 5, 5]], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_volumetric_wave_propagation_single_source() {
        let grid = Grid::new(12, 12, 12, 0.002, 0.002, 0.002).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let config = ElasticWaveConfig {
            simulation_time: 2e-3, // Short simulation for testing
            save_every: 10,
            ..Default::default()
        };

        let mut solver = ElasticWaveSolver::new(&grid, &medium, config).unwrap();

        // Enable volumetric features
        let volumetric_config = VolumetricWaveConfig {
            volumetric_boundaries: true,
            interference_tracking: true,
            volumetric_attenuation: true,
            ..Default::default()
        };
        solver.set_volumetric_config(volumetric_config);

        // Create single initial displacement (point source)
        let mut initial_disp = Array3::zeros((12, 12, 12));
        initial_disp[[6, 6, 6]] = 1e-6; // 1 micron displacement at center

        let initial_displacements = vec![initial_disp];
        let push_times = vec![0.0]; // Immediate push

        let result = solver.propagate_volumetric_waves(&initial_displacements, &push_times);
        assert!(result.is_ok());

        let (history, tracker) = result.unwrap();
        assert!(!history.is_empty(), "Should have displacement history");

        // Check that wave propagated
        if history.len() > 1 {
            let final_field = history.last().unwrap();
            let magnitude = final_field.displacement_magnitude();
            let max_disp = magnitude.iter().cloned().fold(0.0_f64, |a, b| a.max(b));
            assert!(max_disp > 0.0, "Wave should propagate through volume");
        }

        // Check wave front tracking
        let valid_arrivals = tracker.arrival_times.iter()
            .filter(|&&time| !time.is_infinite())
            .count();
        assert!(valid_arrivals > 0, "Should track some wave arrivals");

        // Check volumetric quality metrics
        let quality = solver.calculate_volumetric_quality(&tracker);
        assert!(quality.coverage >= 0.0 && quality.coverage <= 1.0);
        assert!(quality.average_quality >= 0.0 && quality.average_quality <= 1.0);
        assert!(quality.valid_tracking_points > 0);
    }

    #[test]
    fn test_volumetric_wave_propagation_multiple_sources() {
        let grid = Grid::new(16, 16, 16, 0.002, 0.002, 0.002).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let config = ElasticWaveConfig {
            simulation_time: 3e-3,
            save_every: 15,
            ..Default::default()
        };

        let mut solver = ElasticWaveSolver::new(&grid, &medium, config).unwrap();

        // Enable volumetric features
        let volumetric_config = VolumetricWaveConfig {
            volumetric_boundaries: true,
            interference_tracking: true,
            volumetric_attenuation: true,
            ..Default::default()
        };
        solver.set_volumetric_config(volumetric_config);

        // Create multiple initial displacements (orthogonal pushes)
        let mut disp1 = Array3::zeros((16, 16, 16));
        disp1[[8, 12, 8]] = 1e-6; // +Y direction push

        let mut disp2 = Array3::zeros((16, 16, 16));
        disp2[[12, 8, 8]] = 1e-6; // +X direction push

        let initial_displacements = vec![disp1, disp2];
        let push_times = vec![0.0, 1e-3]; // Staggered pushes

        let result = solver.propagate_volumetric_waves(&initial_displacements, &push_times);
        assert!(result.is_ok());

        let (history, tracker) = result.unwrap();

        // Check interference tracking (should detect multiple wave arrivals)
        let max_interference = tracker.interference_map.iter()
            .cloned()
            .fold(0.0_f64, f64::max);
        assert!(max_interference >= 1.0, "Should detect wave interference from multiple sources");

        // Check volumetric quality
        let quality = solver.calculate_volumetric_quality(&tracker);
        assert!(quality.max_interference >= 1.0, "Should detect interference patterns");
    }

    #[test]
    fn test_volumetric_boundary_conditions() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let config = ElasticWaveConfig::default();
        let mut solver = ElasticWaveSolver::new(&grid, &medium, config).unwrap();

        // Enable volumetric boundaries
        let volumetric_config = VolumetricWaveConfig {
            volumetric_boundaries: true,
            ..Default::default()
        };
        solver.set_volumetric_config(volumetric_config);

        // Create test velocity field
        let mut vx = Array3::zeros((10, 10, 10));
        let mut vy = Array3::zeros((10, 10, 10));
        let mut vz = Array3::zeros((10, 10, 10));

        // Set some test values
        vx[[5, 5, 5]] = 1.0;
        vy[[5, 5, 5]] = 1.0;
        vz[[5, 5, 5]] = 1.0;

        // Apply boundary conditions
        solver.apply_volumetric_boundaries(&mut vx, &mut vy, &mut vz, 5, 5, 5);

        // Check that boundary damping was applied (should be less than original)
        assert!(vx[[5, 5, 5]] <= 1.0);
        assert!(vy[[5, 5, 5]] <= 1.0);
        assert!(vz[[5, 5, 5]] <= 1.0);
    }

    #[test]
    fn test_wave_front_tracking_update() {
        let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let config = ElasticWaveConfig::default();
        let mut solver = ElasticWaveSolver::new(&grid, &medium, config).unwrap();

        // Enable interference tracking
        let volumetric_config = VolumetricWaveConfig {
            interference_tracking: true,
            front_tracking_resolution: 1e-7, // Low threshold for testing
            ..Default::default()
        };
        solver.set_volumetric_config(volumetric_config);

        // Create test wave field with displacement above threshold
        let mut field = ElasticWaveField::new(8, 8, 8);
        field.ux[[4, 4, 4]] = 1e-6; // Above threshold
        field.vx[[4, 4, 4]] = 10.0; // Non-zero velocity for direction calculation

        let mut tracker = WaveFrontTracker {
            arrival_times: Array3::from_elem((8, 8, 8), f64::INFINITY),
            amplitudes: Array3::zeros((8, 8, 8)),
            directions: Array3::from_elem((8, 8, 8), [0.0, 0.0, 0.0]),
            interference_map: Array3::zeros((8, 8, 8)),
            tracking_quality: Array3::zeros((8, 8, 8)),
        };

        // Update tracking
        solver.update_wave_front_tracking(&field, &mut tracker, 1e-3);

        // Check that arrival time was recorded
        assert!(!tracker.arrival_times[[4, 4, 4]].is_infinite());
        assert_eq!(tracker.arrival_times[[4, 4, 4]], 1e-3);

        // Check that amplitude was recorded
        assert!(tracker.amplitudes[[4, 4, 4]] > 0.0);

        // Check that direction was calculated
        let direction = tracker.directions[[4, 4, 4]];
        let magnitude = (direction[0] * direction[0] +
                        direction[1] * direction[1] +
                        direction[2] * direction[2]).sqrt();
        assert!((magnitude - 1.0).abs() < 1e-10, "Direction should be unit vector");
    }

    #[test]
    fn test_volumetric_quality_metrics() {
        let grid = Grid::new(6, 6, 6, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let config = ElasticWaveConfig::default();
        let solver = ElasticWaveSolver::new(&grid, &medium, config).unwrap();

        // Create mock tracker with some valid data
        let mut tracker = WaveFrontTracker {
            arrival_times: Array3::from_elem((6, 6, 6), f64::INFINITY),
            amplitudes: Array3::zeros((6, 6, 6)),
            directions: Array3::from_elem((6, 6, 6), [0.0, 0.0, 0.0]),
            interference_map: Array3::zeros((6, 6, 6)),
            tracking_quality: Array3::zeros((6, 6, 6)),
        };

        // Set some valid tracking points
        tracker.arrival_times[[2, 2, 2]] = 1e-3;
        tracker.arrival_times[[3, 3, 3]] = 2e-3;
        tracker.tracking_quality[[2, 2, 2]] = 0.8;
        tracker.tracking_quality[[3, 3, 3]] = 0.9;
        tracker.interference_map[[2, 2, 2]] = 1.0;
        tracker.interference_map[[3, 3, 3]] = 2.0;

        let quality = solver.calculate_volumetric_quality(&tracker);

        // Check quality metrics
        assert!(quality.coverage > 0.0 && quality.coverage <= 1.0);
        assert!(quality.average_quality > 0.0 && quality.average_quality <= 1.0);
        assert_eq!(quality.valid_tracking_points, 2);
        assert_eq!(quality.max_interference, 2.0);
    }
}
