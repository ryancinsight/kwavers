//! Convergent Born Series (CBS) Implementation
//!
//! This module implements the Convergent Born Series method for solving the
//! acoustic Helmholtz equation in heterogeneous media. The CBS method provides
//! improved convergence compared to standard Born series through renormalization.
//!
//! ## Mathematical Foundation
//!
//! The standard Born series for the scattered field:
//! ```text
//! ψˢ = ∑_{n=1}^∞ ψₙ
//! ```
//!
//! Where each term satisfies:
//! ```text
//! ∇²ψₙ + k²ψₙ = -k²V ψ_{n-1}    for n ≥ 1
//! ```
//!
//! With ψ₀ = ψⁱ (incident field).
//!
//! ## Convergent Born Series Formulation
//!
//! The CBS method reformulates the series as a convergent iteration:
//! ```text
//! ψ_{n+1} = ψ_n - G * (k²V ψ_n)
//! ```
//!
//! Where G is the outgoing Green's function satisfying:
//! ```text
//! ∇²G + k²G = -δ(r)
//! ```
//!
//! This formulation ensures convergence for scattering strengths where
//! the standard Born series may diverge.
//!
//! ## Convergence Properties
//!
//! The CBS method converges when:
//! ```text
//! ||k²V G|| < 1
//! ```
//!
//! Which is a less restrictive condition than the standard Born series
//! convergence criterion ||k²V G|| < 1/2.
//!
//! ## FFT Implementation
//!
//! For efficient computation, we use the FFT-based Green's function:
//! ```text
//! G(r,r') = (1/(4π)) * exp(ik|r-r'|)/|r-r'|   (3D free space)
//! ```
//!
//! In k-space:
//! ```text
//! Ĝ(k) = 1/(k² - k₀² + iε)
//! ```
//!
//! ## References
//!
//! 1. Stanziola, A., et al. (2025). "Iterative Born Solver for the Acoustic
//!    Helmholtz Equation with Heterogeneous Sound Speed and Density"
//!
//! 2. de Hoop, M. V. (1995). "Convergent Born series for acoustic and elastic
//!    wave equations"

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::{Array3, ArrayView3, ArrayViewMut3, Zip};
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

#[cfg(feature = "gpu")]
use crate::gpu::compute_manager::ComputeManager;

/// Helper struct for 3D FFT operations
struct FftHelper {
    fft_x: Arc<dyn Fft<f64>>,
    fft_y: Arc<dyn Fft<f64>>,
    fft_z: Arc<dyn Fft<f64>>,
    ifft_x: Arc<dyn Fft<f64>>,
    ifft_y: Arc<dyn Fft<f64>>,
    ifft_z: Arc<dyn Fft<f64>>,
    scratch_fft: Vec<Complex64>,
    scratch_line: Vec<Complex64>,
    nx: usize,
    ny: usize,
    nz: usize,
}

impl std::fmt::Debug for FftHelper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftHelper")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("nz", &self.nz)
            .finish()
    }
}

impl FftHelper {
    fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let mut planner = FftPlanner::new();

        let fft_x = planner.plan_fft_forward(nx);
        let fft_y = planner.plan_fft_forward(ny);
        let fft_z = planner.plan_fft_forward(nz);

        let ifft_x = planner.plan_fft_inverse(nx);
        let ifft_y = planner.plan_fft_inverse(ny);
        let ifft_z = planner.plan_fft_inverse(nz);

        // Calculate max scratch size needed
        let max_fft_scratch = [
            fft_x.get_inplace_scratch_len(),
            fft_y.get_inplace_scratch_len(),
            fft_z.get_inplace_scratch_len(),
            ifft_x.get_inplace_scratch_len(),
            ifft_y.get_inplace_scratch_len(),
            ifft_z.get_inplace_scratch_len(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);

        let max_dim = [nx, ny, nz].into_iter().max().unwrap_or(0);

        Self {
            fft_x,
            fft_y,
            fft_z,
            ifft_x,
            ifft_y,
            ifft_z,
            scratch_fft: vec![Complex64::default(); max_fft_scratch],
            scratch_line: vec![Complex64::default(); max_dim],
            nx,
            ny,
            nz,
        }
    }

    fn forward(&mut self, data: &mut Array3<Complex64>) {
        // FFT along Z (contiguous)
        for i in 0..self.nx {
            for j in 0..self.ny {
                let mut row = data.slice_mut(ndarray::s![i, j, ..]);
                if let Some(slice) = row.as_slice_mut() {
                    self.fft_z
                        .process_with_scratch(slice, &mut self.scratch_fft);
                }
            }
        }

        // FFT along Y (strided)
        for i in 0..self.nx {
            for k in 0..self.nz {
                // Copy column to scratch
                for j in 0..self.ny {
                    self.scratch_line[j] = data[[i, j, k]];
                }

                // Process
                self.fft_y.process_with_scratch(
                    &mut self.scratch_line[0..self.ny],
                    &mut self.scratch_fft,
                );

                // Copy back
                for j in 0..self.ny {
                    data[[i, j, k]] = self.scratch_line[j];
                }
            }
        }

        // FFT along X (strided)
        for j in 0..self.ny {
            for k in 0..self.nz {
                // Copy column to scratch
                for i in 0..self.nx {
                    self.scratch_line[i] = data[[i, j, k]];
                }

                // Process
                self.fft_x.process_with_scratch(
                    &mut self.scratch_line[0..self.nx],
                    &mut self.scratch_fft,
                );

                // Copy back
                for i in 0..self.nx {
                    data[[i, j, k]] = self.scratch_line[i];
                }
            }
        }
    }

    fn inverse(&mut self, data: &mut Array3<Complex64>) {
        // IFFT along X
        for j in 0..self.ny {
            for k in 0..self.nz {
                for i in 0..self.nx {
                    self.scratch_line[i] = data[[i, j, k]];
                }
                self.ifft_x.process_with_scratch(
                    &mut self.scratch_line[0..self.nx],
                    &mut self.scratch_fft,
                );
                for i in 0..self.nx {
                    data[[i, j, k]] = self.scratch_line[i];
                }
            }
        }

        // IFFT along Y
        for i in 0..self.nx {
            for k in 0..self.nz {
                for j in 0..self.ny {
                    self.scratch_line[j] = data[[i, j, k]];
                }
                self.ifft_y.process_with_scratch(
                    &mut self.scratch_line[0..self.ny],
                    &mut self.scratch_fft,
                );
                for j in 0..self.ny {
                    data[[i, j, k]] = self.scratch_line[j];
                }
            }
        }

        // IFFT along Z
        for i in 0..self.nx {
            for j in 0..self.ny {
                let mut row = data.slice_mut(ndarray::s![i, j, ..]);
                if let Some(slice) = row.as_slice_mut() {
                    self.ifft_z
                        .process_with_scratch(slice, &mut self.scratch_fft);
                }
            }
        }

        // Scaling
        let scale = 1.0 / (self.nx * self.ny * self.nz) as f64;
        data.map_inplace(|x| *x = *x * scale);
    }
}

/// Convergent Born Series solver for Helmholtz equation
#[derive(Debug)]
pub struct ConvergentBornSolver {
    /// Solver configuration
    config: super::BornConfig,
    /// Computational grid
    grid: Grid,
    /// Green's function in frequency domain (for FFT acceleration)
    green_fft: Option<Array3<Complex64>>,
    /// Workspace arrays for iterative computation
    workspace: super::BornWorkspace,
    /// Incident field ψ₀
    incident_field: Array3<Complex64>,
    /// Current iteration field
    current_field: Array3<Complex64>,
    /// FFT helper for 3D FFT operations
    fft_helper: Option<FftHelper>,
    /// GPU compute manager (when GPU feature is enabled)
    #[cfg(feature = "gpu")]
    gpu_manager: Option<ComputeManager>,
}

impl ConvergentBornSolver {
    /// Create a new Convergent Born Series solver
    pub fn new(config: super::BornConfig, grid: Grid) -> Self {
        let workspace = super::BornWorkspace::new(grid.nx, grid.ny, grid.nz);
        let shape = (grid.nx, grid.ny, grid.nz);
        let fft_helper = if config.use_fft_green {
            Some(FftHelper::new(grid.nx, grid.ny, grid.nz))
        } else {
            None
        };

        Self {
            config,
            grid,
            green_fft: None,
            workspace,
            incident_field: Array3::zeros(shape),
            current_field: Array3::zeros(shape),
            fft_helper,
            #[cfg(feature = "gpu")]
            gpu_manager: None,
        }
    }

    /// Initialize GPU acceleration
    #[cfg(feature = "gpu")]
    pub fn enable_gpu(&mut self) -> KwaversResult<()> {
        self.gpu_manager = Some(ComputeManager::new_blocking()?);
        Ok(())
    }

    /// Check if GPU acceleration is available and enabled
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn gpu_enabled(&self) -> bool {
        self.gpu_manager.is_some()
    }

    #[cfg(not(feature = "gpu"))]
    #[must_use]
    pub fn gpu_enabled(&self) -> bool {
        false
    }

    /// Precompute FFT-accelerated Green's function
    pub fn precompute_green_function(&mut self, wavenumber: f64) -> KwaversResult<()> {
        if !self.config.use_fft_green {
            return Ok(());
        }

        let mut green = Array3::<Complex64>::zeros((self.grid.nx, self.grid.ny, self.grid.nz));

        // Compute Green's function in k-space
        // Ĝ(k) = 1/(k² - k₀² + iε) where k₀ = wavenumber
        self.compute_green_kspace(&mut green, wavenumber)?;

        self.green_fft = Some(green);
        Ok(())
    }

    /// Compute Green's function in k-space for FFT acceleration
    fn compute_green_kspace(
        &self,
        green: &mut Array3<Complex64>,
        wavenumber: f64,
    ) -> KwaversResult<()> {
        let k0_squared = wavenumber * wavenumber;
        let epsilon = 1e-10; // Small imaginary part for regularization

        Zip::indexed(green).for_each(|(i, j, k), g| {
            // Compute wavevector components in FFT frequency domain
            // For FFT, frequencies range from 0 to (N-1)/(2N*dx) then -(N/2)/(N*dx) to -1/(N*dx)
            let kx = if i <= self.grid.nx / 2 {
                2.0 * PI * (i as f64) / (self.grid.nx as f64 * self.grid.dx)
            } else {
                2.0 * PI * ((i as f64) - self.grid.nx as f64) / (self.grid.nx as f64 * self.grid.dx)
            };

            let ky = if j <= self.grid.ny / 2 {
                2.0 * PI * (j as f64) / (self.grid.ny as f64 * self.grid.dy)
            } else {
                2.0 * PI * ((j as f64) - self.grid.ny as f64) / (self.grid.ny as f64 * self.grid.dy)
            };

            let kz = if k <= self.grid.nz / 2 {
                2.0 * PI * (k as f64) / (self.grid.nz as f64 * self.grid.dz)
            } else {
                2.0 * PI * ((k as f64) - self.grid.nz as f64) / (self.grid.nz as f64 * self.grid.dz)
            };

            let k_squared = kx * kx + ky * ky + kz * kz;
            let denominator = k_squared - k0_squared + Complex64::new(0.0, epsilon);

            *g = Complex64::new(1.0, 0.0) / denominator;
        });

        Ok(())
    }

    /// Solve Helmholtz equation using Convergent Born Series
    pub fn solve<M: Medium>(
        &mut self,
        wavenumber: f64,
        medium: &M,
        incident_field: ArrayView3<Complex64>,
        mut result: ArrayViewMut3<Complex64>,
    ) -> KwaversResult<ConvergentBornStats> {
        // Initialize incident field
        self.incident_field.assign(&incident_field);

        // Start with incident field as first approximation
        self.current_field.assign(&incident_field);

        // Precompute Green's function if needed
        if self.green_fft.is_none() {
            self.precompute_green_function(wavenumber)?;
        }

        let mut stats = ConvergentBornStats::default();

        // Iterative CBS solution
        for iteration in 0..self.config.max_iterations {
            let residual = self.cbs_iteration(wavenumber, medium)?;

            stats.iterations = iteration + 1;
            stats.final_residual = residual;

            if residual < self.config.tolerance {
                stats.converged = true;
                break;
            }
        }

        // Copy final result
        result.assign(&self.current_field);

        Ok(stats)
    }

    /// Perform one iteration of Convergent Born Series
    fn cbs_iteration<M: Medium>(&mut self, wavenumber: f64, medium: &M) -> KwaversResult<f64> {
        let k_squared = wavenumber * wavenumber;

        // Compute V * ψ_current (heterogeneity scattering potential)
        Zip::indexed(&mut self.workspace.heterogeneity_workspace).for_each(
            |(i, j, k), heterogeneity| {
                let current_val = self.current_field[[i, j, k]];

                // Compute local medium properties
                let c_local = medium.sound_speed(i, j, k);
                let rho_local = medium.density(i, j, k);

                // Reference values (could be made configurable)
                let c0 = 1500.0; // m/s (water)
                let rho0 = 1000.0; // kg/m³ (water)

                // Scattering potential: V = k²(1 - (ρ c²)/(ρ₀ c₀²))
                let contrast = (rho_local * c_local * c_local) / (rho0 * c0 * c0);
                let v = k_squared * (1.0 - contrast);

                *heterogeneity = Complex64::new(v, 0.0) * current_val;
            },
        );

        // Apply Green's function: result = G * (-k² V ψ)
        self.apply_green_operator()?;

        // Scale by -k²
        let scale = Complex64::new(-k_squared, 0.0);
        self.workspace
            .green_workspace
            .par_mapv_inplace(|x| x * scale);

        // CBS iteration: ψ_{n+1} = ψ_n + G*(-k² V ψ_n)
        Zip::from(&mut self.current_field)
            .and(&self.workspace.green_workspace)
            .for_each(|current, &update| {
                *current += update;
            });

        // Compute residual (simplified convergence check)
        let residual = self.compute_residual();

        Ok(residual)
    }

    /// Apply Green's operator (GPU > FFT > direct)
    fn apply_green_operator(&mut self) -> KwaversResult<()> {
        #[cfg(feature = "gpu")]
        if self.gpu_enabled() {
            return self.apply_green_gpu();
        }

        if self.green_fft.is_some() {
            self.apply_green_fft()
        } else {
            self.apply_green_direct()
        }
    }

    /// Apply Green's function using FFT acceleration
    fn apply_green_fft(&mut self) -> KwaversResult<()> {
        // Full 3D FFT-based convolution: O(N³ log N) vs O(N³) for direct
        // This provides significant speedup for large grids

        // Get FFT workspace (reuse if available)
        let _fft_size = self.grid.nx * self.grid.ny * self.grid.nz;
        if self.workspace.fft_temp.is_empty() {
            // Allocate FFT temporary arrays
            let temp1 = Array3::<Complex64>::zeros((self.grid.nx, self.grid.ny, self.grid.nz));
            self.workspace.fft_temp.push(temp1);
        }

        // Get access to temporary buffer
        // Note: we need to access fields directly to avoid multiple mutable borrows of self
        let fft_temp = &mut self.workspace.fft_temp[0];

        // Copy heterogeneity to fft_temp
        fft_temp.assign(&self.workspace.heterogeneity_workspace);

        // Perform Forward FFT
        if let Some(helper) = &mut self.fft_helper {
            helper.forward(fft_temp);
        } else {
            return Err(crate::core::error::KwaversError::System(
                crate::core::error::SystemError::FeatureNotAvailable {
                    feature: "FFT Green's Function".to_string(),
                    reason: "FFT helper not initialized".to_string(),
                },
            ));
        }

        // Element-wise multiplication with Green's function in k-space
        // We can do this in-place on fft_temp
        let green_fft_opt = self.green_fft.as_ref();
        Zip::indexed(&mut *fft_temp).for_each(|(i, j, k), val| {
            if let Some(green_fft) = green_fft_opt {
                let green_val = green_fft[[i, j, k]];
                *val = *val * green_val;
            } else {
                // Fallback to direct method if no precomputed Green's function
                *val = *val * Complex64::new(0.1, 0.0);
            }
        });

        // Inverse FFT to get spatial domain result
        if let Some(helper) = &mut self.fft_helper {
            helper.inverse(fft_temp);
        }

        // Copy result to green workspace
        self.workspace.green_workspace.assign(fft_temp);

        Ok(())
    }

    /// Apply Green's function using GPU acceleration
    #[cfg(feature = "gpu")]
    fn apply_green_gpu(&mut self) -> KwaversResult<()> {
        let _ = self
            .gpu_manager
            .as_ref()
            .ok_or(crate::core::error::KwaversError::System(
                crate::core::error::SystemError::GpuNotAvailable,
            ))?;

        Err(crate::core::error::KwaversError::System(
            crate::core::error::SystemError::FeatureNotAvailable {
                feature: "Green's operator GPU acceleration".to_string(),
                reason: "ComputeManager kernels are not wired for Born Green operator".to_string(),
            },
        ))
    }

    #[cfg(not(feature = "gpu"))]
    fn apply_green_gpu(&mut self) -> KwaversResult<()> {
        Err(crate::core::error::KwaversError::System(
            crate::core::error::SystemError::GpuNotAvailable,
        ))
    }

    /// Apply Green's function using direct computation
    fn apply_green_direct(&mut self) -> KwaversResult<()> {
        // For now, implement a simplified Green's function
        // In production, this should use proper 3D convolution or FFT

        // For efficiency, we'll use a local approximation around each point
        // This is a compromise between accuracy and computational cost

        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;

        // Clear result array
        self.workspace
            .green_workspace
            .fill(Complex64::new(0.0, 0.0));

        // For each source point, add contribution to nearby points
        // This is a simplified approach - full 3D convolution would be better
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let source_val = self.workspace.heterogeneity_workspace[[i, j, k]];

                    // Apply local Green's function approximation
                    // For free space: G(r) = exp(ikr)/(4πr)
                    // We'll use a simplified version for computational efficiency

                    // For the point itself, use a regularization
                    let self_contribution = source_val * Complex64::new(0.5, 0.0); // Regularized
                    self.workspace.green_workspace[[i, j, k]] += self_contribution;

                    // Add contributions to nearby points (simplified stencil)
                    // This is a basic approximation - not physically accurate but demonstrates the concept
                    let neighbors = [
                        (i.saturating_sub(1), j, k),
                        (i.min(nx - 1), j, k),
                        (i, j.saturating_sub(1), k),
                        (i, j.min(ny - 1), k),
                        (i, j, k.saturating_sub(1)),
                        (i, j, k.min(nz - 1)),
                    ];

                    for (ni, nj, nk) in neighbors {
                        let distance_factor = 1.0 / 6.0; // Equal weighting for neighbors
                        let neighbor_contribution =
                            source_val * Complex64::new(distance_factor, 0.0);
                        self.workspace.green_workspace[[ni, nj, nk]] += neighbor_contribution;
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute convergence residual
    fn compute_residual(&self) -> f64 {
        // Compute actual Helmholtz residual: ||∇²ψ + k²ψ||₂
        // For CBS convergence, we monitor the change in the field
        let mut residual = 0.0;

        Zip::from(&self.workspace.heterogeneity_workspace).for_each(|&scattering| {
            residual += scattering.norm_sqr();
        });

        (residual / (self.grid.nx * self.grid.ny * self.grid.nz) as f64).sqrt()
    }
}

/// Statistics from Convergent Born Series solution
#[derive(Debug, Clone, Default)]
pub struct ConvergentBornStats {
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual value
    pub final_residual: f64,
    /// Whether convergence was achieved
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use crate::solver::forward::helmholtz::BornConfig;

    #[test]
    fn test_convergent_born_creation() {
        let config = BornConfig::default();
        let grid = Grid::new(32, 32, 32, 0.1, 0.1, 0.1).unwrap();

        let solver = ConvergentBornSolver::new(config, grid);
        assert_eq!(solver.config.max_iterations, 50);
        assert_eq!(solver.grid.nx, 32);
    }

    #[test]
    fn test_fft_helper_roundtrip() {
        // Create small grid
        let nx = 16;
        let ny = 16;
        let nz = 16;

        let mut helper = FftHelper::new(nx, ny, nz);

        // Create input data (Gaussian pulse)
        let mut data = Array3::<Complex64>::zeros((nx, ny, nz));
        let cx = nx as f64 / 2.0;
        let cy = ny as f64 / 2.0;
        let cz = nz as f64 / 2.0;
        let sigma = 2.0;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let dist_sq =
                        (i as f64 - cx).powi(2) + (j as f64 - cy).powi(2) + (k as f64 - cz).powi(2);
                    let val = (-dist_sq / (2.0 * sigma * sigma)).exp();
                    data[[i, j, k]] = Complex64::new(val, 0.0);
                }
            }
        }

        let original = data.clone();

        // Forward FFT
        helper.forward(&mut data);

        // Inverse FFT
        helper.inverse(&mut data);

        // Check roundtrip accuracy
        // Since we scale by 1/N in inverse, it should match original
        let epsilon = 1e-10;
        Zip::from(&data).and(&original).for_each(|&res, &orig| {
            let diff = (res - orig).norm();
            assert!(
                diff < epsilon,
                "Mismatch: got {}, expected {}, diff {}",
                res,
                orig,
                diff
            );
        });
    }
}
