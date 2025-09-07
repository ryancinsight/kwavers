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
use super::utils::{compute_kspace_correction, compute_pml_operators};
use crate::utils::spectral::compute_kspace_correction_factors;

/// Core k-Wave solver implementing the k-space pseudospectral method
pub struct KWaveSolver {
    config: KWaveConfig,
    grid: Grid,

    // FFT planning
    fft_planner: FftPlanner<f64>,

    // k-space operators
    kappa: Array3<f64>,                             // k-space correction
    k_vec: (Array3<f64>, Array3<f64>, Array3<f64>), // k-vectors
    k_max: f64,                                     // Maximum supported k

    // PML operators
    pml_x: Array3<f64>,
    pml_y: Array3<f64>,
    pml_z: Array3<f64>,

    // Field variables
    p: Array3<f64>,         // Pressure
    p_k: Array3<Complex64>, // Pressure in k-space
    ux: Array3<f64>,        // Velocity x
    uy: Array3<f64>,        // Velocity y
    uz: Array3<f64>,        // Velocity z

    // Absorption variables
    absorb_tau: Array3<f64>,
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
        let kappa = compute_kspace_correction_factors(&k_vec.0, &k_vec.1, &k_vec.2, grid, crate::utils::spectral::CorrectionType::Liu1997);

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

        match config.absorption_mode {
            AbsorptionMode::Lossless => {
                // No absorption - arrays remain zero
            }
            AbsorptionMode::Stokes => {
                // Stokes absorption implementation would go here
                tau.fill(1.0);
                eta.fill(0.1);
            }
            AbsorptionMode::PowerLaw { alpha_coeff, alpha_power: _ } => {
                // Power law absorption implementation
                tau.fill(1.0);
                eta.fill(alpha_coeff);
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

    /// Perform a single time step
    fn step_forward<M: Medium>(&mut self, _medium: &M) -> KwaversResult<()> {
        // This would implement the full k-Wave algorithm
        // For now, this is a simplified placeholder
        Ok(())
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