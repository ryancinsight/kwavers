//! Unified acoustic wave solver architecture
//!
//! This module provides a unified interface for all acoustic wave models,
//! eliminating code duplication and ensuring consistent implementations.

use crate::{
    error::{KwaversError, KwaversResult, ValidationError},
    fft::{Fft3d, Ifft3d},
    grid::Grid,
    medium::Medium,
};
use ndarray::{Array3, Array4, Axis, Zip};
use std::sync::{Arc, Mutex};
use std::time::Instant;

type Complex<T> = num_complex::Complex<T>;

/// Type of acoustic model to use
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AcousticModelType {
    /// Linear acoustic wave equation
    Linear,
    /// Westervelt equation for nonlinear acoustics in thermoviscous fluids
    Westervelt,
    /// Kuznetsov equation for nonlinear acoustics with diffusion
    Kuznetsov,
}

/// Configuration for the unified acoustic solver
#[derive(Debug, Clone)]
pub struct AcousticSolverConfig {
    /// Type of acoustic model to use
    pub model_type: AcousticModelType,

    /// Enable k-space correction for numerical dispersion
    pub k_space_correction: bool,

    /// Order of k-space correction (1 or 2)
    pub k_space_order: usize,

    /// Nonlinearity scaling factor (B/A parameter related)
    pub nonlinearity_scaling: f64,

    /// Maximum allowed pressure for stability
    pub max_pressure: f64,

    /// CFL safety factor for stability
    pub cfl_safety_factor: f64,

    /// Enable adaptive time stepping
    pub use_adaptive_timestep: bool,

    /// Source frequency for frequency-dependent effects [Hz]
    pub source_frequency: f64,

    /// Acoustic diffusivity for Kuznetsov model [m²/s]
    pub acoustic_diffusivity: f64,
}

impl Default for AcousticSolverConfig {
    fn default() -> Self {
        Self {
            model_type: AcousticModelType::Linear,
            k_space_correction: true,
            k_space_order: 2,
            nonlinearity_scaling: 1.0,
            max_pressure: 1e8, // 100 MPa
            cfl_safety_factor: 0.3,
            use_adaptive_timestep: false,
            source_frequency: 1e6,        // 1 MHz
            acoustic_diffusivity: 4.5e-6, // Water at 20°C
        }
    }
}

/// Performance metrics for the solver
#[derive(Debug, Default, Clone)]
pub struct PerformanceMetrics {
    pub call_count: u64,
    pub nonlinear_time: f64,
    pub fft_time: f64,
    pub source_time: f64,
    pub total_time: f64,
}

/// Unified acoustic wave solver
///
/// This solver consolidates the implementations of Linear, Westervelt, and Kuznetsov
/// acoustic models into a single, consistent architecture.
pub struct AcousticWaveSolver {
    /// Configuration for the solver
    config: AcousticSolverConfig,

    /// Grid for spatial discretization
    grid: Grid,

    /// Precomputed k-space arrays
    k_squared: Array3<f64>,
    kx: Array3<f64>,
    ky: Array3<f64>,
    kz: Array3<f64>,

    /// Pressure history for time derivatives (Westervelt)
    pressure_history: Option<Array3<f64>>,
    prev_pressure: Option<Array3<f64>>,

    /// RK4 workspace for Kuznetsov integration
    rk4_workspace: Option<Array4<f64>>,

    /// Performance metrics
    metrics: Arc<Mutex<PerformanceMetrics>>,

    /// Cached maximum sound speed for stability checks
    max_sound_speed: f64,
}

impl AcousticWaveSolver {
    /// Create a new unified acoustic wave solver
    pub fn new(config: AcousticSolverConfig, grid: Grid) -> KwaversResult<Self> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

        // Precompute k-space arrays using efficient ndarray operations
        let kx_vec = grid.kx();
        let ky_vec = grid.ky();
        let kz_vec = grid.kz();

        // Use Zip for efficient computation
        let mut k_squared = Array3::<f64>::zeros((nx, ny, nz));
        let mut kx = Array3::<f64>::zeros((nx, ny, nz));
        let mut ky = Array3::<f64>::zeros((nx, ny, nz));
        let mut kz = Array3::<f64>::zeros((nx, ny, nz));

        Zip::indexed(&mut k_squared)
            .and(&mut kx)
            .and(&mut ky)
            .and(&mut kz)
            .for_each(|(i, j, k), k_sq, kx_val, ky_val, kz_val| {
                let kx_i = kx_vec[i];
                let ky_j = ky_vec[j];
                let kz_k = kz_vec[k];
                *kx_val = kx_i;
                *ky_val = ky_j;
                *kz_val = kz_k;
                *k_sq = kx_i.powi(2) + ky_j.powi(2) + kz_k.powi(2);
            });

        // Initialize model-specific storage
        let pressure_history = match config.model_type {
            AcousticModelType::Westervelt => Some(Array3::zeros((nx, ny, nz))),
            _ => None,
        };

        let prev_pressure = match config.model_type {
            AcousticModelType::Westervelt => Some(Array3::zeros((nx, ny, nz))),
            _ => None,
        };

        let rk4_workspace = match config.model_type {
            AcousticModelType::Kuznetsov => Some(Array4::zeros((4, nx, ny, nz))),
            _ => None,
        };

        Ok(Self {
            config,
            grid,
            k_squared,
            kx,
            ky,
            kz,
            pressure_history,
            prev_pressure,
            rk4_workspace,
            metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
            max_sound_speed: 1500.0, // Will be updated from medium
        })
    }

    /// Update the acoustic wave field
    pub fn update_wave(
        &mut self,
        pressure: &mut Array3<f64>,
        medium: &dyn Medium,
        source_term: &Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        let start = Instant::now();

        // Update cached sound speed if needed
        self.update_max_sound_speed(medium);

        // Check stability
        self.check_stability(dt)?;

        // Dispatch to appropriate model
        match self.config.model_type {
            AcousticModelType::Linear => self.update_linear(pressure, medium, source_term, dt)?,
            AcousticModelType::Westervelt => {
                self.update_westervelt(pressure, medium, source_term, dt)?
            }
            AcousticModelType::Kuznetsov => {
                self.update_kuznetsov(pressure, medium, source_term, dt)?
            }
        }

        // Update metrics
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.call_count += 1;
            metrics.total_time += start.elapsed().as_secs_f64();
        }

        Ok(())
    }

    /// Linear acoustic wave equation update
    fn update_linear(
        &mut self,
        pressure: &mut Array3<f64>,
        medium: &dyn Medium,
        source_term: &Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        let start_fft = Instant::now();

        // Transform to k-space
        let mut pressure_complex = Array3::<Complex<f64>>::zeros(pressure.dim());
        Zip::from(&mut pressure_complex)
            .and(pressure.view())
            .for_each(|c, &r| *c = Complex::new(r, 0.0));

        let mut fft = Fft3d::new(self.grid.nx, self.grid.ny, self.grid.nz);
        fft.process(&mut pressure_complex, &self.grid);

        // Apply k-space propagation with dispersion correction
        if self.config.k_space_correction {
            self.apply_kspace_correction(&mut pressure_complex, medium, dt)?;
        }

        // Transform back to real space
        let mut ifft = Ifft3d::new(self.grid.nx, self.grid.ny, self.grid.nz);
        ifft.process(&mut pressure_complex, &self.grid);

        // Extract real part and add source term
        Zip::from(pressure)
            .and(&pressure_complex)
            .and(source_term)
            .for_each(|p, c, &s| *p = c.re + s * dt);

        // Update FFT time metric
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.fft_time += start_fft.elapsed().as_secs_f64();
        }

        Ok(())
    }

    /// Westervelt equation update (formerly ViscoelasticWave)
    fn update_westervelt(
        &mut self,
        pressure: &mut Array3<f64>,
        medium: &dyn Medium,
        source_term: &Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        // Store pressure history for second-order time derivative
        if let Some(ref mut prev) = self.prev_pressure {
            if let Some(ref mut history) = self.pressure_history {
                history.assign(prev);
            }
            prev.assign(pressure);
        }

        // Compute nonlinear term: (β/ρ₀c₀⁴) * ∂²p²/∂t²
        let nonlinear_term = self.compute_westervelt_nonlinearity(pressure, medium, dt)?;

        // Linear propagation in k-space (do this after computing nonlinear term to avoid borrow issues)
        // For now, inline the linear update to avoid borrow checker issues
        let start_fft = Instant::now();

        // Transform to k-space
        let mut pressure_complex = Array3::<Complex<f64>>::zeros(pressure.dim());
        Zip::from(&mut pressure_complex)
            .and(pressure.view())
            .for_each(|c, &r| *c = Complex::new(r, 0.0));

        let mut fft = Fft3d::new(self.grid.nx, self.grid.ny, self.grid.nz);
        fft.process(&mut pressure_complex, &self.grid);

        // Apply k-space propagation with dispersion correction
        if self.config.k_space_correction {
            self.apply_kspace_correction(&mut pressure_complex, medium, dt)?;
        }

        // Transform back to real space
        let mut ifft = Ifft3d::new(self.grid.nx, self.grid.ny, self.grid.nz);
        ifft.process(&mut pressure_complex, &self.grid);

        // Extract real part and add both source and nonlinear terms
        Zip::from(pressure)
            .and(&pressure_complex)
            .and(source_term)
            .and(&nonlinear_term)
            .for_each(|p, c, &s, &nl| *p = c.re + s * dt + nl * dt);

        // Update FFT time metric
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.fft_time += start_fft.elapsed().as_secs_f64();
        }

        Ok(())
    }

    /// Kuznetsov equation update
    fn update_kuznetsov(
        &mut self,
        pressure: &mut Array3<f64>,
        medium: &dyn Medium,
        source_term: &Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        // Clone pressure to avoid borrow issues
        let pressure_clone = pressure.clone();

        // Compute all RK4 stages first (before borrowing workspace mutably)
        // k1 = f(y_n, t_n)
        let k1 = self.compute_kuznetsov_rhs(&pressure_clone, medium, source_term)?;

        // k2 = f(y_n + dt/2 * k1, t_n + dt/2)
        let mut temp = pressure_clone.clone();
        Zip::from(&mut temp)
            .and(&k1)
            .for_each(|t, &k| *t += 0.5 * dt * k);
        let k2 = self.compute_kuznetsov_rhs(&temp, medium, source_term)?;

        // k3 = f(y_n + dt/2 * k2, t_n + dt/2)
        temp.assign(&pressure_clone);
        Zip::from(&mut temp)
            .and(&k2)
            .for_each(|t, &k| *t += 0.5 * dt * k);
        let k3 = self.compute_kuznetsov_rhs(&temp, medium, source_term)?;

        // k4 = f(y_n + dt * k3, t_n + dt)
        temp.assign(&pressure_clone);
        Zip::from(&mut temp).and(&k3).for_each(|t, &k| *t += dt * k);
        let k4 = self.compute_kuznetsov_rhs(&temp, medium, source_term)?;

        // Store in workspace if available (for potential debugging/analysis)
        if let Some(workspace) = self.rk4_workspace.as_mut() {
            workspace.index_axis_mut(Axis(0), 0).assign(&k1);
            workspace.index_axis_mut(Axis(0), 1).assign(&k2);
            workspace.index_axis_mut(Axis(0), 2).assign(&k3);
            workspace.index_axis_mut(Axis(0), 3).assign(&k4);
        }

        // Update: y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        Zip::from(pressure)
            .and(&k1)
            .and(&k2)
            .and(&k3)
            .and(&k4)
            .for_each(|p, &k1_val, &k2_val, &k3_val, &k4_val| {
                *p += dt / 6.0 * (k1_val + 2.0 * k2_val + 2.0 * k3_val + k4_val);
            });

        Ok(())
    }

    /// Compute Westervelt nonlinearity term
    fn compute_westervelt_nonlinearity(
        &self,
        pressure: &Array3<f64>,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<Array3<f64>> {
        // Requires pressure history for ∂²p²/∂t²
        if let (Some(ref prev), Some(ref history)) = (&self.prev_pressure, &self.pressure_history) {
            let density = medium.density_array();
            let sound_speed = medium.sound_speed_array();
            let beta = self.config.nonlinearity_scaling;

            let mut nonlinear = Array3::zeros(pressure.dim());

            // Compute ∂²p²/∂t² using finite differences
            Zip::from(&mut nonlinear)
                .and(pressure)
                .and(prev)
                .and(history)
                .and(density)
                .and(sound_speed)
                .for_each(|nl, &p, &p_prev, &p_hist, &rho, &c| {
                    let p_squared = p * p;
                    let p_prev_squared = p_prev * p_prev;
                    let p_hist_squared = p_hist * p_hist;
                    let d2p2_dt2 = (p_squared - 2.0 * p_prev_squared + p_hist_squared) / (dt * dt);
                    *nl = (beta / (rho * c.powi(4))) * d2p2_dt2;
                });

            Ok(nonlinear)
        } else {
            Ok(Array3::zeros(pressure.dim()))
        }
    }

    /// Compute Kuznetsov equation right-hand side
    fn compute_kuznetsov_rhs(
        &self,
        pressure: &Array3<f64>,
        medium: &dyn Medium,
        source_term: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let density = medium.density_array();
        let sound_speed = medium.sound_speed_array();

        // Compute Laplacian in k-space
        let mut pressure_complex = Array3::<Complex<f64>>::zeros(pressure.dim());
        Zip::from(&mut pressure_complex)
            .and(pressure)
            .for_each(|c, &r| *c = Complex::new(r, 0.0));

        let mut fft = Fft3d::new(self.grid.nx, self.grid.ny, self.grid.nz);
        fft.process(&mut pressure_complex, &self.grid);

        let mut laplacian_hat = pressure_complex.clone();
        Zip::from(&mut laplacian_hat)
            .and(&self.k_squared)
            .for_each(|l, &k2| *l *= -k2);

        let mut ifft = Ifft3d::new(self.grid.nx, self.grid.ny, self.grid.nz);
        ifft.process(&mut laplacian_hat, &self.grid);

        let mut laplacian = Array3::zeros(pressure.dim());
        Zip::from(&mut laplacian)
            .and(&laplacian_hat)
            .for_each(|l, c| *l = c.re);

        // Compute gradient for nonlinear term
        let grad = self.compute_gradient_spectral(pressure)?;

        let mut rhs = Array3::zeros(pressure.dim());

        // Combine terms: c²∇²p + β/(2ρc²) * |∇p|² + δ∇²(∂p/∂t) + source
        Zip::from(&mut rhs)
            .and(&laplacian)
            .and(&grad)
            .and(density)
            .and(sound_speed)
            .and(source_term)
            .for_each(|r, &lap, &g, &rho, &c, &s| {
                let linear = c * c * lap;
                let nonlinear = self.config.nonlinearity_scaling / (2.0 * rho * c * c) * g * g;
                let diffusion = self.config.acoustic_diffusivity * lap;
                *r = linear + nonlinear + diffusion + s;
            });

        Ok(rhs)
    }

    /// Compute gradient using spectral method
    fn compute_gradient_spectral(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let mut field_complex = Array3::<Complex<f64>>::zeros(field.dim());
        Zip::from(&mut field_complex)
            .and(field)
            .for_each(|c, &r| *c = Complex::new(r, 0.0));

        let mut fft = Fft3d::new(self.grid.nx, self.grid.ny, self.grid.nz);
        fft.process(&mut field_complex, &self.grid);

        // Compute ∇p in k-space
        let mut grad_x_hat = field_complex.clone();
        let mut grad_y_hat = field_complex.clone();
        let mut grad_z_hat = field_complex.clone();

        Zip::from(&mut grad_x_hat)
            .and(&self.kx)
            .for_each(|g, &k| *g *= Complex::new(0.0, k));
        Zip::from(&mut grad_y_hat)
            .and(&self.ky)
            .for_each(|g, &k| *g *= Complex::new(0.0, k));
        Zip::from(&mut grad_z_hat)
            .and(&self.kz)
            .for_each(|g, &k| *g *= Complex::new(0.0, k));

        // Transform back
        let mut ifft = Ifft3d::new(self.grid.nx, self.grid.ny, self.grid.nz);
        ifft.process(&mut grad_x_hat, &self.grid);
        ifft.process(&mut grad_y_hat, &self.grid);
        ifft.process(&mut grad_z_hat, &self.grid);

        // Extract real parts
        let mut grad_x = Array3::zeros(field.dim());
        let mut grad_y = Array3::zeros(field.dim());
        let mut grad_z = Array3::zeros(field.dim());

        Zip::from(&mut grad_x)
            .and(&grad_x_hat)
            .for_each(|g, c| *g = c.re);
        Zip::from(&mut grad_y)
            .and(&grad_y_hat)
            .for_each(|g, c| *g = c.re);
        Zip::from(&mut grad_z)
            .and(&grad_z_hat)
            .for_each(|g, c| *g = c.re);

        let mut grad_magnitude = Array3::zeros(field.dim());
        Zip::from(&mut grad_magnitude)
            .and(&grad_x)
            .and(&grad_y)
            .and(&grad_z)
            .for_each(|g, &gx, &gy, &gz| {
                *g = (gx * gx + gy * gy + gz * gz).sqrt();
            });

        Ok(grad_magnitude)
    }

    /// Apply k-space correction for numerical dispersion
    ///
    /// # Important Note on Heterogeneous Media
    ///
    /// In heterogeneous media, the k-space correction factor is spatially dependent.
    /// The current implementation uses a global correction based on the maximum sound speed,
    /// which is conservative but may introduce phase errors. For more accurate results
    /// in strongly heterogeneous media, consider using a split-step method that alternates
    /// between k-space propagation and real-space medium corrections.
    fn apply_kspace_correction(
        &self,
        pressure_hat: &mut Array3<Complex<f64>>,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        // Use maximum sound speed for conservative stability
        let c_max = self.max_sound_speed;
        let dx = self.grid.dx;

        // Compute correction factor based on k-space order
        let correction_factor = match self.config.k_space_order {
            1 => {
                // First-order correction: sinc(kΔx/2)
                Zip::from(pressure_hat)
                    .and(&self.k_squared)
                    .for_each(|p, &k2| {
                        let k = k2.sqrt();
                        let arg = k * dx / 2.0;
                        let sinc = if arg.abs() < 1e-10 {
                            1.0
                        } else {
                            arg.sin() / arg
                        };
                        *p *= sinc.powi(2);
                    });
            }
            2 => {
                // Second-order correction with CFL-based scaling
                let cfl = c_max * dt / dx;
                Zip::from(pressure_hat)
                    .and(&self.k_squared)
                    .for_each(|p, &k2| {
                        let correction = (-k2 * c_max * c_max * dt * dt / 2.0).exp();
                        *p *= correction * (1.0 - cfl * cfl / 6.0);
                    });
            }
            _ => {
                // No correction
            }
        };

        Ok(())
    }

    /// Update cached maximum sound speed from medium
    fn update_max_sound_speed(&mut self, medium: &dyn Medium) {
        let sound_speed = medium.sound_speed_array();
        self.max_sound_speed = sound_speed.iter().fold(0.0f64, |max, &c| max.max(c));
    }

    /// Check CFL stability condition
    fn check_stability(&self, dt: f64) -> KwaversResult<()> {
        let dx = self.grid.dx;
        let cfl = self.max_sound_speed * dt / dx;
        let max_cfl = self.config.cfl_safety_factor;

        if cfl > max_cfl {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "CFL".to_string(),
                value: format!("{}", cfl),
                constraint: format!(
                    "must be <= {} (c_max={}, dt={}, dx={})",
                    max_cfl, self.max_sound_speed, dt, dx
                ),
            }));
        }

        Ok(())
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.lock().map(|m| m.clone()).unwrap_or_default()
    }

    /// Reset performance metrics
    pub fn reset_metrics(&mut self) {
        if let Ok(mut metrics) = self.metrics.lock() {
            *metrics = PerformanceMetrics::default();
        }
    }
}
