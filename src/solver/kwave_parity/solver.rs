//! k-Wave Core Solver Implementation
//!
//! Main solver implementation following GRASP principles.
//! This module focuses solely on the core solving logic.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::Array3;
use rustfft::{num_complex::Complex64, FftPlanner};

use super::config::{AbsorptionMode, KWaveConfig};
use super::data::{FieldArrays, KSpaceData};
use super::operators::kspace::compute_k_operators;
use super::utils::compute_pml_operators;
use crate::utils::spectral::compute_kspace_correction_factors;

/// Core k-Wave solver implementing the k-space pseudospectral method
pub struct KWaveSolver {
    #[allow(dead_code)]
    config: KWaveConfig,
    #[allow(dead_code)]
    grid: Grid,

    // FFT planning
    #[allow(dead_code)]
    fft_planner: FftPlanner<f64>,

    // k-space operators
    #[allow(dead_code)]
    kappa: Array3<f64>, // k-space correction
    #[allow(dead_code)]
    k_vec: (Array3<f64>, Array3<f64>, Array3<f64>), // k-vectors
    k_max: f64, // Maximum supported k

    // PML operators
    #[allow(dead_code)]
    pml_x: Array3<f64>,
    #[allow(dead_code)]
    pml_y: Array3<f64>,
    #[allow(dead_code)]
    pml_z: Array3<f64>,

    // Field variables
    p: Array3<f64>, // Pressure
    #[allow(dead_code)]
    p_k: Array3<Complex64>, // Pressure in k-space
    ux: Array3<f64>, // Velocity x
    uy: Array3<f64>, // Velocity y
    uz: Array3<f64>, // Velocity z

    // Absorption variables
    #[allow(dead_code)]
    absorb_tau: Array3<f64>,
    #[allow(dead_code)]
    absorb_eta: Array3<f64>,
}

impl std::fmt::Debug for KWaveSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KWaveSolver")
            .field("config", &self.config)
            .field("grid", &"Grid { ... }")
            .field("k_max", &self.k_max)
            .finish()
    }
}

impl KWaveSolver {
    /// Create a new k-Wave solver
    pub fn new(config: KWaveConfig, grid: Grid) -> KwaversResult<Self> {
        let (kspace_data, k_max) = Self::initialize_kspace_operators(&grid)?;
        let pml_operators = Self::initialize_pml_operators(&config, &grid);
        let field_arrays = Self::initialize_field_arrays(&grid);
        let absorption_operators = Self::initialize_absorption_operators(&config, &grid, k_max);

        Ok(Self {
            config,
            grid,
            fft_planner: FftPlanner::new(),
            kappa: kspace_data.kappa,
            k_vec: kspace_data.k_vec,
            k_max,
            pml_x: pml_operators.0,
            pml_y: pml_operators.1,
            pml_z: pml_operators.2,
            p: field_arrays.p,
            p_k: field_arrays.p_k,
            ux: field_arrays.ux,
            uy: field_arrays.uy,
            uz: field_arrays.uz,
            absorb_tau: absorption_operators.0,
            absorb_eta: absorption_operators.1,
        })
    }

    /// Initialize k-space operators and correction factors
    fn initialize_kspace_operators(grid: &Grid) -> KwaversResult<(KSpaceData, f64)> {
        let (k_ops, k_max) = compute_k_operators(grid);
        let k_vec = (k_ops.kx.clone(), k_ops.ky.clone(), k_ops.kz.clone());
        let kappa = compute_kspace_correction_factors(
            &k_vec.0,
            &k_vec.1,
            &k_vec.2,
            grid,
            crate::utils::spectral::CorrectionType::Liu1997,
        );

        Ok((KSpaceData { kappa, k_vec }, k_max))
    }

    /// Initialize PML boundary operators
    fn initialize_pml_operators(
        config: &KWaveConfig,
        grid: &Grid,
    ) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let pml_x = compute_pml_operators(grid, config.pml_size, config.pml_alpha);
        let pml_y = pml_x.clone(); // Same for all directions in this implementation
        let pml_z = pml_x.clone();
        (pml_x, pml_y, pml_z)
    }

    /// Initialize field arrays
    fn initialize_field_arrays(grid: &Grid) -> FieldArrays {
        FieldArrays {
            p: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            p_k: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            ux: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            uy: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            uz: Array3::zeros((grid.nx, grid.ny, grid.nz)),
        }
    }

    /// Initialize absorption operators
    fn initialize_absorption_operators(
        config: &KWaveConfig,
        grid: &Grid,
        _k_max: f64,
    ) -> (Array3<f64>, Array3<f64>) {
        let mut tau = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut eta = Array3::zeros((grid.nx, grid.ny, grid.nz));

        match &config.absorption_mode {
            AbsorptionMode::Lossless => {
                // No absorption - arrays remain zero
            }
            AbsorptionMode::Stokes => {
                // Stokes absorption implementation would go here
                tau.fill(1.0);
                eta.fill(0.1);
            }
            AbsorptionMode::PowerLaw {
                alpha_coeff,
                alpha_power: _,
            } => {
                // Power law absorption implementation
                tau.fill(1.0);
                eta.fill(*alpha_coeff);
            }
            AbsorptionMode::MultiRelaxation {
                tau: rel_tau,
                weights,
            } => {
                // Multi-relaxation absorption - compute effective coefficients
                let mut effective_tau = 0.0;
                let mut effective_eta = 0.0;
                for (&tau_val, &weight) in rel_tau.iter().zip(weights.iter()) {
                    effective_tau += weight * tau_val;
                    effective_eta += weight * tau_val * tau_val;
                }
                tau.fill(effective_tau);
                eta.fill(effective_eta);
            }
            AbsorptionMode::Causal {
                relaxation_times,
                alpha_0,
            } => {
                // Causal absorption with multiple relaxation times
                let mut effective_tau = 0.0;
                let mut effective_eta = 0.0;
                for &tau_val in relaxation_times.iter() {
                    let contribution = alpha_0 / (relaxation_times.len() as f64);
                    effective_tau += contribution * tau_val;
                    effective_eta += contribution * tau_val * tau_val;
                }
                tau.fill(effective_tau);
                eta.fill(effective_eta);
            }
        }

        (tau, eta)
    }

    /// Run the k-Wave simulation
    pub fn run<M: Medium>(&mut self, medium: &M, steps: usize) -> KwaversResult<()> {
        for _step in 0..steps {
            self.step_forward(medium)?;
        }
        Ok(())
    }

    /// Perform a single time step using k-space pseudospectral method
    ///
    /// Implements the k-Wave algorithm following Treeby & Cox (2010):
    /// 1. Compute spatial derivatives in k-space (spectral accuracy)
    /// 2. Update particle velocities from pressure gradients
    /// 3. Update pressure from velocity divergence
    /// 4. Apply absorption using fractional Laplacian
    /// 5. Apply PML boundary conditions
    ///
    /// References:
    /// - Treeby & Cox (2010) "k-Wave: MATLAB toolbox" J. Biomed. Opt. 15(2)
    /// - Liu (1997) "Weighted essentially non-oscillatory schemes" J. Comput. Phys.
    fn step_forward<M: Medium>(&mut self, medium: &M) -> KwaversResult<()> {
        use super::nonlinearity::update_pressure_with_nonlinearity;
        use crate::utils::fft_operations::{fft_3d_array, ifft_3d_array};
        use ndarray::Zip;

        let dt = self.config.dt;
        let (nx, ny, nz) = self.p.dim();

        // Step 1: Transform pressure to k-space for spectral derivatives
        let p_k = fft_3d_array(&self.p);

        // Step 2: Compute pressure gradients in k-space (spectral accuracy)
        // ∂p/∂x = ifft(ikx · fft(p))
        let mut dpx_k = Array3::zeros((nx, ny, nz));
        let mut dpy_k = Array3::zeros((nx, ny, nz));
        let mut dpz_k = Array3::zeros((nx, ny, nz));

        Zip::from(&mut dpx_k)
            .and(&p_k)
            .and(&self.k_vec.0)
            .for_each(|dpx, &pk, &kx| {
                *dpx = Complex64::new(0.0, kx) * pk;
            });

        Zip::from(&mut dpy_k)
            .and(&p_k)
            .and(&self.k_vec.1)
            .for_each(|dpy, &pk, &ky| {
                *dpy = Complex64::new(0.0, ky) * pk;
            });

        Zip::from(&mut dpz_k)
            .and(&p_k)
            .and(&self.k_vec.2)
            .for_each(|dpz, &pk, &kz| {
                *dpz = Complex64::new(0.0, kz) * pk;
            });

        // Transform gradients back to physical space
        let dpx = ifft_3d_array(&dpx_k);
        let dpy = ifft_3d_array(&dpy_k);
        let dpz = ifft_3d_array(&dpz_k);

        // Step 3: Update particle velocities using momentum equation
        // ρ₀ ∂u/∂t = -∇p
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);
                    let rho0 = crate::medium::density_at(medium, x, y, z, &self.grid);

                    self.ux[[i, j, k]] -= (dt / rho0) * dpx[[i, j, k]];
                    self.uy[[i, j, k]] -= (dt / rho0) * dpy[[i, j, k]];
                    self.uz[[i, j, k]] -= (dt / rho0) * dpz[[i, j, k]];
                }
            }
        }

        // Step 4: Compute velocity divergence in k-space
        // ∇·u = ∂ux/∂x + ∂uy/∂y + ∂uz/∂z
        let ux_k = fft_3d_array(&self.ux);
        let uy_k = fft_3d_array(&self.uy);
        let uz_k = fft_3d_array(&self.uz);

        let mut div_u_k = Array3::zeros((nx, ny, nz));
        let i = Complex64::new(0.0, 1.0);

        // Compute divergence term by term to avoid Zip limitation
        for kk in 0..nz {
            for jj in 0..ny {
                for ii in 0..nx {
                    let kx = self.k_vec.0[[ii, jj, kk]];
                    let ky = self.k_vec.1[[ii, jj, kk]];
                    let kz = self.k_vec.2[[ii, jj, kk]];

                    div_u_k[[ii, jj, kk]] = i
                        * (kx * ux_k[[ii, jj, kk]]
                            + ky * uy_k[[ii, jj, kk]]
                            + kz * uz_k[[ii, jj, kk]]);
                }
            }
        }

        let div_u = ifft_3d_array(&div_u_k);

        // Step 5: Update pressure using continuity equation with nonlinearity
        if self.config.nonlinearity {
            update_pressure_with_nonlinearity(&mut self.p, &div_u, medium, &self.grid, dt)?;
        } else {
            // Linear case: ∂p/∂t = -ρ₀c² ∇·u
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);
                        let rho0 = crate::medium::density_at(medium, x, y, z, &self.grid);
                        let c0 = crate::medium::sound_speed_at(medium, x, y, z, &self.grid);

                        self.p[[i, j, k]] -= dt * rho0 * c0 * c0 * div_u[[i, j, k]];
                    }
                }
            }
        }

        // Step 6: Apply absorption using fractional Laplacian if enabled
        if let super::config::AbsorptionMode::PowerLaw { alpha_power, .. } = self.config.absorption_mode {
            use super::absorption::apply_power_law_absorption;
            apply_power_law_absorption(
                &mut self.p,
                &self.absorb_tau,
                &self.absorb_eta,
                dt,
                &self.k_vec,
                alpha_power,
            )?;
        }

        // Step 7: Apply PML boundary conditions to absorb outgoing waves
        self.apply_pml_damping();

        Ok(())
    }

    /// Apply PML damping to boundaries
    fn apply_pml_damping(&mut self) {
        use ndarray::Zip;

        // Apply exponential damping in PML regions
        Zip::from(&mut self.p)
            .and(&self.pml_x)
            .and(&self.pml_y)
            .and(&self.pml_z)
            .for_each(|p, &pml_x, &pml_y, &pml_z| {
                // Combine PML effects from all boundaries
                let pml_factor = pml_x * pml_y * pml_z;
                *p *= pml_factor;
            });

        // Also damp velocities
        Zip::from(&mut self.ux)
            .and(&self.pml_x)
            .and(&self.pml_y)
            .and(&self.pml_z)
            .for_each(|u, &pml_x, &pml_y, &pml_z| {
                *u *= pml_x * pml_y * pml_z;
            });

        Zip::from(&mut self.uy)
            .and(&self.pml_x)
            .and(&self.pml_y)
            .and(&self.pml_z)
            .for_each(|u, &pml_x, &pml_y, &pml_z| {
                *u *= pml_x * pml_y * pml_z;
            });

        Zip::from(&mut self.uz)
            .and(&self.pml_x)
            .and(&self.pml_y)
            .and(&self.pml_z)
            .for_each(|u, &pml_x, &pml_y, &pml_z| {
                *u *= pml_x * pml_y * pml_z;
            });
    }

    /// Get current pressure field
    pub fn pressure_field(&self) -> &Array3<f64> {
        &self.p
    }

    /// Get current velocity fields
    pub fn velocity_fields(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
        (&self.ux, &self.uy, &self.uz)
    }
}
