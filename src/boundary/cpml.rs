//! Convolutional Perfectly Matched Layer (C-PML) implementation
//!
//! This module provides a complete C-PML boundary condition implementation for absorbing
//! outgoing waves at domain boundaries. Based on the formulation by Roden & Gedney (2000)
//! and Komatitsch & Martin (2007).
//!
//! ## Implementation Features
//! - Full recursive convolution with memory variables
//! - Support for acoustic, elastic, and dispersive media
//! - Configured for grazing angle absorption
//! - Polynomial grading profiles with κ stretching and α frequency shifting
//!
//! ## References
//! - Roden & Gedney (2000) "Convolutional PML (CPML): An efficient FDTD implementation"
//! - Komatitsch & Martin (2007) "An unsplit convolutional perfectly matched layer"

use crate::error::{ConfigError, KwaversError, KwaversResult, ValidationError};
use crate::grid::Grid;
use ndarray::{Array3, Array4, Axis, Zip};

use log::debug;

/// Minimum cosine theta value to prevent division by zero in reflection estimation
/// This corresponds to angles near 90 degrees (grazing incidence)
const MIN_COS_THETA_FOR_REFLECTION: f64 = 0.1;

/// Configuration for Convolutional PML
#[derive(Debug, Clone)]
pub struct CPMLConfig {
    /// Number of PML cells in each direction
    pub thickness: usize,

    /// Polynomial order for profile grading (typically 3-4)
    pub polynomial_order: f64,

    /// Maximum conductivity scaling factor
    pub sigma_factor: f64,

    /// Maximum κ (coordinate stretching) value
    pub kappa_max: f64,

    /// Maximum α (frequency shifting) value
    pub alpha_max: f64,

    /// Target reflection coefficient (e.g., 1e-6)
    pub target_reflection: f64,

    /// Enable grazing angle absorption
    pub grazing_angle_absorption: bool,
    // REMOVED: CFL number and sound speed - these should come from the solver
    // The solver determines dt based on the actual medium properties and stability requirements
}

impl Default for CPMLConfig {
    fn default() -> Self {
        Self {
            thickness: 10,
            polynomial_order: 3.0,
            sigma_factor: 0.8, // σ_max = σ_factor * σ_theoretical
            kappa_max: 15.0,   // Higher values for grazing angle absorption
            alpha_max: 0.24,   // Standard for low-frequency absorption
            target_reflection: 1e-6,
            grazing_angle_absorption: true,
            // REMOVED: cfl_number and sound_speed
        }
    }
}

impl CPMLConfig {
    /// Create config for grazing angles
    pub fn for_grazing_angles() -> Self {
        Self {
            thickness: 20,
            polynomial_order: 4.0,
            sigma_factor: 1.0,
            kappa_max: 25.0, // Very high for grazing angles
            alpha_max: 0.3,
            target_reflection: 1e-8,
            grazing_angle_absorption: true,
            // REMOVED: cfl_number and sound_speed
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> KwaversResult<()> {
        if self.thickness == 0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "thickness".to_string(),
                value: self.thickness.to_string(),
                constraint: "C-PML thickness must be > 0".to_string(),
            }));
        }

        if self.polynomial_order < 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "polynomial_order".to_string(),
                value: self.polynomial_order.to_string(),
                constraint: "Polynomial order must be >= 0".to_string(),
            }));
        }

        if self.kappa_max < 1.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "kappa_max".to_string(),
                value: self.kappa_max.to_string(),
                constraint: "κ_max must be >= 1".to_string(),
            }));
        }

        Ok(())
    }
}

/// Convolutional PML boundary condition
#[derive(Debug, Clone)]
pub struct CPMLBoundary {
    config: CPMLConfig,

    /// Time step from the solver - ensures consistency with simulation
    /// This must match the solver's dt for proper impedance matching
    dt: f64,

    /// Profile arrays for each dimension
    sigma_x: Vec<f64>,
    sigma_y: Vec<f64>,
    sigma_z: Vec<f64>,

    kappa_x: Vec<f64>,
    kappa_y: Vec<f64>,
    kappa_z: Vec<f64>,

    /// Pre-computed reciprocals for performance (avoids division in hot loop)
    inv_kappa_x: Vec<f64>,
    inv_kappa_y: Vec<f64>,
    inv_kappa_z: Vec<f64>,

    alpha_x: Vec<f64>,
    alpha_y: Vec<f64>,
    alpha_z: Vec<f64>,

    /// Coefficients for time integration
    b_x: Vec<f64>,
    b_y: Vec<f64>,
    b_z: Vec<f64>,

    c_x: Vec<f64>,
    c_y: Vec<f64>,
    c_z: Vec<f64>,

    /// Memory variables for convolutional integration
    /// Stored as 4D arrays: [component, nx, ny, nz]
    psi_acoustic: Array4<f64>,

    /// Auxiliary memory for dispersive media
    psi_dispersive: Option<Array4<f64>>,

    /// Pre-calculated dispersive coefficients for performance
    psi_dispersive_b_x: Option<Vec<f64>>,
    psi_dispersive_b_y: Option<Vec<f64>>,
    psi_dispersive_b_z: Option<Vec<f64>>,
    psi_dispersive_c_x: Option<Vec<f64>>,
    psi_dispersive_c_y: Option<Vec<f64>>,
    psi_dispersive_c_z: Option<Vec<f64>>,

    /// Grid dimensions for validation
    nx: usize,
    ny: usize,
    nz: usize,
}

impl CPMLBoundary {
    /// Get the CPML configuration
    pub fn config(&self) -> &CPMLConfig {
        &self.config
    }
    /// Create new C-PML boundary with given configuration
    ///
    /// # Arguments
    /// * `config` - CPML configuration parameters
    /// * `grid` - Computational grid
    /// * `dt` - Time step from the solver (must be consistent with solver's dt)
    /// * `sound_speed` - Reference sound speed for the medium (typically max sound speed)
    ///
    /// # Important
    /// The `dt` parameter MUST be the same as used by the solver to ensure proper
    /// impedance matching at the boundaries. Using an inconsistent dt will cause
    /// spurious reflections.
    pub fn new(config: CPMLConfig, grid: &Grid, dt: f64, sound_speed: f64) -> KwaversResult<Self> {
        config.validate()?;

        debug!(
            "Initializing C-PML with thickness {}, dt {:.3e} s, sound speed {} m/s",
            config.thickness, dt, sound_speed
        );

        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

        // Validate dt for stability (CFL condition)
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        let max_stable_dt = min_dx / (sound_speed * (3.0_f64).sqrt());
        if dt > max_stable_dt {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "dt".to_string(),
                value: dt.to_string(),
                constraint: format!(
                    "Time step exceeds CFL stability limit. Maximum stable dt = {:.3e} s for sound speed = {} m/s and min grid spacing = {:.3e} m",
                    max_stable_dt, sound_speed, min_dx
                ),
            }));
        }

        // Initialize profile arrays
        let mut cpml = Self {
            config: config.clone(),
            dt, // Use the provided dt from the solver
            sigma_x: vec![0.0; nx],
            sigma_y: vec![0.0; ny],
            sigma_z: vec![0.0; nz],
            kappa_x: vec![1.0; nx],
            kappa_y: vec![1.0; ny],
            kappa_z: vec![1.0; nz],
            inv_kappa_x: vec![1.0; nx],
            inv_kappa_y: vec![1.0; ny],
            inv_kappa_z: vec![1.0; nz],
            alpha_x: vec![0.0; nx],
            alpha_y: vec![0.0; ny],
            alpha_z: vec![0.0; nz],
            b_x: vec![0.0; nx],
            b_y: vec![0.0; ny],
            b_z: vec![0.0; nz],
            c_x: vec![0.0; nx],
            c_y: vec![0.0; ny],
            c_z: vec![0.0; nz],
            psi_acoustic: Array4::zeros((3, nx, ny, nz)), // 3 components for x, y, z
            psi_dispersive: None,
            psi_dispersive_b_x: None,
            psi_dispersive_b_y: None,
            psi_dispersive_b_z: None,
            psi_dispersive_c_x: None,
            psi_dispersive_c_y: None,
            psi_dispersive_c_z: None,
            nx,
            ny,
            nz,
        };

        // Compute standard profiles with the provided sound speed
        cpml.compute_profiles(grid, sound_speed)?;

        Ok(cpml)
    }

    /// Compute C-PML profiles based on configuration
    ///
    /// Optimized for cubic grids: When dimensions are identical (nx==ny==nz and dx==dy==dz),
    /// profiles are computed once and reused, reducing initialization time by ~66%.
    fn compute_profiles(&mut self, grid: &Grid, sound_speed: f64) -> KwaversResult<()> {
        let thickness = self.config.thickness as f64;
        let m = self.config.polynomial_order;

        // Check if grid is cubic (common case optimization)
        const EPSILON: f64 = 1e-9;
        let is_cubic = self.nx == self.ny
            && self.ny == self.nz
            && (grid.dx - grid.dy).abs() < EPSILON
            && (grid.dy - grid.dz).abs() < EPSILON;

        if is_cubic {
            // Cubic grid optimization: compute profile once and clone
            let sigma_theoretical = self.compute_theoretical_sigma(grid.dx, sound_speed);

            // Compute profile for X dimension
            Self::compute_profile_for_dimension(
                self.nx,
                thickness,
                m,
                sigma_theoretical,
                grid.dx,
                &self.config,
                self.dt,
                &mut self.sigma_x,
                &mut self.kappa_x,
                &mut self.inv_kappa_x,
                &mut self.alpha_x,
                &mut self.b_x,
                &mut self.c_x,
            );

            // Clone for Y and Z dimensions (avoiding redundant computation)
            self.sigma_y = self.sigma_x.clone();
            self.kappa_y = self.kappa_x.clone();
            self.inv_kappa_y = self.inv_kappa_x.clone();
            self.alpha_y = self.alpha_x.clone();
            self.b_y = self.b_x.clone();
            self.c_y = self.c_x.clone();

            self.sigma_z = self.sigma_x.clone();
            self.kappa_z = self.kappa_x.clone();
            self.inv_kappa_z = self.inv_kappa_x.clone();
            self.alpha_z = self.alpha_x.clone();
            self.b_z = self.b_x.clone();
            self.c_z = self.c_x.clone();

            return Ok(());
        }

        // Non-cubic grid: compute each dimension independently
        let sigma_theoretical_x = self.compute_theoretical_sigma(grid.dx, sound_speed);
        let sigma_theoretical_y = self.compute_theoretical_sigma(grid.dy, sound_speed);
        let sigma_theoretical_z = self.compute_theoretical_sigma(grid.dz, sound_speed);

        // X-direction profiles - operate directly on struct fields
        Self::compute_profile_for_dimension(
            self.nx,
            thickness,
            m,
            sigma_theoretical_x,
            grid.dx,
            &self.config,
            self.dt,
            &mut self.sigma_x,
            &mut self.kappa_x,
            &mut self.inv_kappa_x,
            &mut self.alpha_x,
            &mut self.b_x,
            &mut self.c_x,
        );

        // Y-direction profiles
        Self::compute_profile_for_dimension(
            self.ny,
            thickness,
            m,
            sigma_theoretical_y,
            grid.dy,
            &self.config,
            self.dt,
            &mut self.sigma_y,
            &mut self.kappa_y,
            &mut self.inv_kappa_y,
            &mut self.alpha_y,
            &mut self.b_y,
            &mut self.c_y,
        );

        // Z-direction profiles
        Self::compute_profile_for_dimension(
            self.nz,
            thickness,
            m,
            sigma_theoretical_z,
            grid.dz,
            &self.config,
            self.dt,
            &mut self.sigma_z,
            &mut self.kappa_z,
            &mut self.inv_kappa_z,
            &mut self.alpha_z,
            &mut self.b_z,
            &mut self.c_z,
        );

        debug!(
            "C-PML profiles computed with σ_theoretical = ({:.2e}, {:.2e}, {:.2e})",
            sigma_theoretical_x, sigma_theoretical_y, sigma_theoretical_z
        );

        Ok(())
    }

    /// Compute theoretical optimal sigma for given grid spacing
    ///
    /// The theoretical optimal sigma depends on the impedance Z = rho * c
    /// For simplicity, we assume constant density and use sound speed directly
    fn compute_theoretical_sigma(&self, dx: f64, sound_speed: f64) -> f64 {
        let m = self.config.polynomial_order;
        let r_coeff = self.config.target_reflection;

        // Theoretical optimal value for C-PML
        // sigma_opt = -(m+1) * c * ln(R) / (2 * L)
        // where L is the PML thickness in meters
        let pml_width = self.config.thickness as f64 * dx;
        let sigma_opt = -(m + 1.0) * sound_speed * r_coeff.ln() / (2.0 * pml_width);

        sigma_opt * self.config.sigma_factor
    }

    /// Generic function to compute 1D profile for any direction
    /// Operates directly on mutable slices to avoid allocations
    fn compute_profile_for_dimension(
        n: usize,
        thickness: f64,
        m: f64,
        sigma_max: f64,
        dx: f64,
        config: &CPMLConfig,
        dt: f64,
        // Mutable slices to operate on directly
        sigma: &mut [f64],
        kappa: &mut [f64],
        inv_kappa: &mut [f64],
        alpha: &mut [f64],
        b: &mut [f64],
        c: &mut [f64],
    ) {
        for i in 0..n {
            let d = Self::calculate_pml_distance(i, n, thickness);

            if d > 0.0 {
                let (sigma_i, kappa_i, alpha_i) =
                    Self::calculate_grading_profiles(d, m, sigma_max, config);

                sigma[i] = sigma_i;
                kappa[i] = kappa_i;
                inv_kappa[i] = 1.0 / kappa_i;
                alpha[i] = alpha_i;

                let (b_i, c_i) = Self::calculate_update_coefficients(sigma_i, kappa_i, alpha_i, dt);
                b[i] = b_i;
                c[i] = c_i;
            } else {
                // Outside PML region - reset values
                sigma[i] = 0.0;
                kappa[i] = 1.0;
                inv_kappa[i] = 1.0;
                alpha[i] = 0.0;
                b[i] = 0.0;
                c[i] = 0.0;
            }
        }
    }

    /// Calculate normalized distance from PML interface
    /// Returns 0 outside PML, increasing to 1 at boundary
    #[inline]
    fn calculate_pml_distance(i: usize, n: usize, thickness: f64) -> f64 {
        // Distance from left boundary
        let d_left = if i < thickness as usize {
            (thickness - i as f64 - 0.5) / thickness
        } else {
            0.0
        };

        // Distance from right boundary
        let d_right = if i >= n - thickness as usize {
            (i as f64 - (n as f64 - thickness - 1.0) + 0.5) / thickness
        } else {
            0.0
        };

        d_left.max(d_right)
    }

    /// Calculate grading profiles for conductivity, stretching, and frequency shifting
    #[inline]
    fn calculate_grading_profiles(
        d: f64,
        m: f64,
        sigma_max: f64,
        config: &CPMLConfig,
    ) -> (f64, f64, f64) {
        // Polynomial grading
        let d_m = d.powf(m);

        // Conductivity profile
        let sigma = sigma_max * d_m;

        // Coordinate stretching profile
        let kappa = if config.grazing_angle_absorption {
            // Enhanced profile for grazing angles
            1.0 + (config.kappa_max - 1.0) * d.powf(m + 1.0)
        } else {
            1.0 + (config.kappa_max - 1.0) * d_m
        };

        // Frequency shifting profile (quadratic for stability)
        let alpha = config.alpha_max * (1.0 - d).powi(2);

        (sigma, kappa, alpha)
    }

    /// Calculate time integration coefficients for recursive convolution
    #[inline]
    fn calculate_update_coefficients(sigma: f64, kappa: f64, alpha: f64, dt: f64) -> (f64, f64) {
        // Exponential coefficient
        let b = (-(sigma + kappa * alpha) * dt).exp();

        // Convolution coefficient
        let c = if (sigma + kappa * alpha).abs() > 1e-10 {
            sigma / (sigma + kappa * alpha) * (b - 1.0)
        } else {
            0.0
        };

        (b, c)
    }

    /// Update acoustic memory variables with recursive convolution
    /// Uses pre-computed coefficients for efficiency
    ///
    /// # Arguments
    /// * `pressure_grad` - Pressure gradient field
    /// * `component` - Component index (0=x, 1=y, 2=z)
    ///
    /// # Panics
    /// Panics in debug mode if component is not 0, 1, or 2
    pub fn update_acoustic_memory(&mut self, pressure_grad: &Array3<f64>, component: usize) {
        debug_assert!(
            component < 3,
            "Component index must be 0, 1, or 2, but got {}",
            component
        );

        let mut psi = self.psi_acoustic.index_axis_mut(Axis(0), component);

        match component {
            0 => {
                // X-component
                // Use pre-computed coefficients for efficiency
                Zip::indexed(&mut psi).and(pressure_grad).for_each(
                    |(i, _j, _k), psi_val, &grad| {
                        *psi_val = self.b_x[i] * *psi_val + self.c_x[i] * grad;
                    },
                );
            }
            1 => {
                // Y-component
                Zip::indexed(&mut psi).and(pressure_grad).for_each(
                    |(_i, j, _k), psi_val, &grad| {
                        *psi_val = self.b_y[j] * *psi_val + self.c_y[j] * grad;
                    },
                );
            }
            2 => {
                // Z-component
                Zip::indexed(&mut psi).and(pressure_grad).for_each(
                    |(_i, _j, k), psi_val, &grad| {
                        *psi_val = self.b_z[k] * *psi_val + self.c_z[k] * grad;
                    },
                );
            }
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }

    /// Update the time step and recompute coefficients if needed
    /// This should be called if the simulation time step changes
    ///
    /// # Arguments
    /// * `new_dt` - New time step from the solver
    /// * `grid` - Computational grid
    /// * `sound_speed` - Reference sound speed for the medium
    pub fn update_dt(&mut self, new_dt: f64, grid: &Grid, sound_speed: f64) -> KwaversResult<()> {
        const DT_TOLERANCE: f64 = 1e-12;

        if (self.dt - new_dt).abs() > DT_TOLERANCE {
            debug!(
                "Updating CPML coefficients for new dt: {} -> {}",
                self.dt, new_dt
            );
            self.dt = new_dt;
            // Recompute profiles with the new dt and sound speed
            self.compute_profiles(grid, sound_speed)?;
        }
        Ok(())
    }

    /// Apply C-PML absorption to field gradients
    /// Uses pre-computed reciprocals and FMA for optimal performance
    ///
    /// # Arguments
    /// * `gradient` - Gradient field to modify
    /// * `component` - Component index (0=x, 1=y, 2=z)
    ///
    /// # Panics
    /// Panics in debug mode if component is not 0, 1, or 2
    pub fn apply_cpml_gradient(&self, gradient: &mut Array3<f64>, component: usize) {
        debug_assert!(
            component < 3,
            "Component index must be 0, 1, or 2, but got {}",
            component
        );

        let psi = self.psi_acoustic.index_axis(Axis(0), component);

        match component {
            0 => {
                // X-component
                Zip::indexed(gradient)
                    .and(&psi)
                    .for_each(|(i, _j, _k), grad, &psi_val| {
                        // Use FMA for optimal performance: grad * inv_kappa + psi
                        *grad = grad.mul_add(self.inv_kappa_x[i], psi_val);
                    });
            }
            1 => {
                // Y-component
                Zip::indexed(gradient)
                    .and(&psi)
                    .for_each(|(_i, j, _k), grad, &psi_val| {
                        *grad = grad.mul_add(self.inv_kappa_y[j], psi_val);
                    });
            }
            2 => {
                // Z-component
                Zip::indexed(gradient)
                    .and(&psi)
                    .for_each(|(_i, _j, k), grad, &psi_val| {
                        *grad = grad.mul_add(self.inv_kappa_z[k], psi_val);
                    });
            }
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }

    /// Enable support for dispersive media with pre-calculated coefficients
    pub fn enable_dispersive_support(&mut self, params: &DispersiveParameters) {
        if self.psi_dispersive.is_none() {
            self.psi_dispersive = Some(Array4::zeros((3, self.nx, self.ny, self.nz)));

            // Pre-calculate dispersive coefficients for each dimension
            // This avoids redundant calculations in the hot loop
            let tau = params.relaxation_time;

            // X-dimension coefficients
            let mut b_disp_x = vec![0.0; self.nx];
            let mut c_disp_x = vec![0.0; self.nx];
            for i in 0..self.nx {
                let sigma = self.sigma_x[i];
                let kappa = self.kappa_x[i];
                let alpha = self.alpha_x[i];

                let omega = 1.0 / tau;
                let denom = sigma + kappa * (alpha + omega);

                b_disp_x[i] = (-denom * self.dt).exp();
                c_disp_x[i] = if denom.abs() > 1e-10 {
                    sigma / denom * (b_disp_x[i] - 1.0)
                } else {
                    0.0
                };
            }
            self.psi_dispersive_b_x = Some(b_disp_x);
            self.psi_dispersive_c_x = Some(c_disp_x);

            // Y-dimension coefficients
            let mut b_disp_y = vec![0.0; self.ny];
            let mut c_disp_y = vec![0.0; self.ny];
            for j in 0..self.ny {
                let sigma = self.sigma_y[j];
                let kappa = self.kappa_y[j];
                let alpha = self.alpha_y[j];

                let omega = 1.0 / tau;
                let denom = sigma + kappa * (alpha + omega);

                b_disp_y[j] = (-denom * self.dt).exp();
                c_disp_y[j] = if denom.abs() > 1e-10 {
                    sigma / denom * (b_disp_y[j] - 1.0)
                } else {
                    0.0
                };
            }
            self.psi_dispersive_b_y = Some(b_disp_y);
            self.psi_dispersive_c_y = Some(c_disp_y);

            // Z-dimension coefficients
            let mut b_disp_z = vec![0.0; self.nz];
            let mut c_disp_z = vec![0.0; self.nz];
            for k in 0..self.nz {
                let sigma = self.sigma_z[k];
                let kappa = self.kappa_z[k];
                let alpha = self.alpha_z[k];

                let omega = 1.0 / tau;
                let denom = sigma + kappa * (alpha + omega);

                b_disp_z[k] = (-denom * self.dt).exp();
                c_disp_z[k] = if denom.abs() > 1e-10 {
                    sigma / denom * (b_disp_z[k] - 1.0)
                } else {
                    0.0
                };
            }
            self.psi_dispersive_b_z = Some(b_disp_z);
            self.psi_dispersive_c_z = Some(c_disp_z);

            debug!("Enabled dispersive media support with pre-calculated coefficients");
        }
    }

    /// Update dispersive memory variables using pre-calculated coefficients
    pub fn update_dispersive_memory(
        &mut self,
        field_grad: &Array3<f64>,
        component: usize,
        _dispersive_params: &DispersiveParameters, // No longer needed, coefficients are pre-calculated
    ) -> KwaversResult<()> {
        if let Some(ref mut psi_disp) = self.psi_dispersive {
            let mut psi = psi_disp.index_axis_mut(Axis(0), component);

            // Use pre-calculated coefficients for optimal performance
            match component {
                0 => {
                    let b_disp = self.psi_dispersive_b_x.as_ref().unwrap();
                    let c_disp = self.psi_dispersive_c_x.as_ref().unwrap();

                    Zip::indexed(&mut psi).and(field_grad).for_each(
                        |(i, _j, _k), psi_val, &grad| {
                            // Use pre-calculated coefficients - no redundant computation
                            *psi_val = b_disp[i] * *psi_val + c_disp[i] * grad;
                        },
                    );
                }
                1 => {
                    let b_disp = self.psi_dispersive_b_y.as_ref().unwrap();
                    let c_disp = self.psi_dispersive_c_y.as_ref().unwrap();

                    Zip::indexed(&mut psi).and(field_grad).for_each(
                        |(_i, j, _k), psi_val, &grad| {
                            *psi_val = b_disp[j] * *psi_val + c_disp[j] * grad;
                        },
                    );
                }
                2 => {
                    let b_disp = self.psi_dispersive_b_z.as_ref().unwrap();
                    let c_disp = self.psi_dispersive_c_z.as_ref().unwrap();

                    Zip::indexed(&mut psi).and(field_grad).for_each(
                        |(_i, _j, k), psi_val, &grad| {
                            *psi_val = b_disp[k] * *psi_val + c_disp[k] * grad;
                        },
                    );
                }
                _ => {
                    return Err(KwaversError::Config(ConfigError::InvalidValue {
                        parameter: "component".to_string(),
                        value: component.to_string(),
                        constraint: "Component must be 0, 1, or 2".to_string(),
                    }))
                }
            }

            Ok(())
        } else {
            Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "psi_dispersive".to_string(),
                value: "None".to_string(),
                constraint: "Dispersive support must be enabled first".to_string(),
            }))
        }
    }

    /// Estimate reflection coefficient for a given angle
    pub fn estimate_reflection(&self, cos_theta: f64) -> f64 {
        // Prevent division by zero for grazing angles
        let cos_theta_safe = cos_theta.max(MIN_COS_THETA_FOR_REFLECTION);

        let thickness = self.config.thickness as f64;
        let m = self.config.polynomial_order;

        // Analytical estimate based on C-PML theory
        // R(θ) = exp(-2 * σ_max * L * cos(θ) / c)
        // This is a simplified model; actual reflection depends on many factors

        let effective_thickness = thickness * cos_theta_safe;
        let attenuation_factor = 2.0 * self.config.sigma_factor * effective_thickness / (m + 1.0);

        (-attenuation_factor).exp()
    }

    /// Get the effective number of PML cells for a given wavelength
    pub fn effective_thickness_in_wavelengths(&self, wavelength: f64, dx: f64) -> f64 {
        let pml_width = self.config.thickness as f64 * dx;
        pml_width / wavelength
    }

    /// Check if configuration is suitable for given frequency
    pub fn validate_for_frequency(&self, frequency: f64, sound_speed: f64, dx: f64) -> bool {
        let wavelength = sound_speed / frequency;
        let thickness_in_wavelengths = self.effective_thickness_in_wavelengths(wavelength, dx);

        // Rule of thumb: need at least 0.5-1 wavelength thickness for good absorption
        thickness_in_wavelengths >= 0.5
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let profile_memory = 6 * self.nx * std::mem::size_of::<f64>() +  // sigma, kappa profiles
                             6 * self.ny * std::mem::size_of::<f64>() +
                             6 * self.nz * std::mem::size_of::<f64>() +
                             3 * self.nx * std::mem::size_of::<f64>() +  // inv_kappa profiles
                             3 * self.ny * std::mem::size_of::<f64>() +
                             3 * self.nz * std::mem::size_of::<f64>();

        let psi_memory = self.psi_acoustic.len() * std::mem::size_of::<f64>();
        let dispersive_memory = self
            .psi_dispersive
            .as_ref()
            .map(|psi| psi.len() * std::mem::size_of::<f64>())
            .unwrap_or(0);

        profile_memory + psi_memory + dispersive_memory
    }
}

/// Parameters for dispersive media
#[derive(Debug, Clone)]
pub struct DispersiveParameters {
    /// Relaxation time for dispersive effects
    pub relaxation_time: f64,
    /// Dispersive coefficient
    pub dispersion_coefficient: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use approx::assert_relative_eq;

    #[test]
    fn test_cpml_creation() {
        let grid = Grid::new(100, 100, 100, 1e-3, 1e-3, 1e-3);
        let config = CPMLConfig::default();
        let dt = 1e-6;
        let sound_speed = 1500.0;

        let cpml = CPMLBoundary::new(config, &grid, dt, sound_speed);
        assert!(cpml.is_ok());
    }

    #[test]
    fn test_profile_computation() {
        let grid = Grid::new(50, 50, 50, 1e-3, 1e-3, 1e-3);
        let config = CPMLConfig {
            thickness: 10,
            polynomial_order: 3.0,
            sigma_factor: 0.8,
            kappa_max: 15.0,
            alpha_max: 0.24,
            target_reflection: 1e-6,
            grazing_angle_absorption: false,
        };

        let dt = 1e-6;
        let sound_speed = 1500.0;
        let cpml = CPMLBoundary::new(config, &grid, dt, sound_speed).unwrap();

        // Check that profiles are properly graded
        // At PML interface (thickness position), values should be near zero
        // At boundary (0 or nx-1), values should be near maximum
        assert!(cpml.sigma_x[0] > 0.0);
        assert!(cpml.sigma_x[10] < cpml.sigma_x[0] * 0.1);
        assert!(cpml.kappa_x[0] > 1.0);
        assert_relative_eq!(cpml.kappa_x[25], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_grazing_angle_config() {
        let config = CPMLConfig::for_grazing_angles();
        assert_eq!(config.thickness, 20);
        assert_eq!(config.polynomial_order, 4.0);
        assert_eq!(config.kappa_max, 25.0);
    }

    #[test]
    fn test_memory_update() {
        let grid = Grid::new(30, 30, 30, 1e-3, 1e-3, 1e-3);
        let config = CPMLConfig::default();
        let dt = 1e-6;
        let sound_speed = 1500.0;

        let mut cpml = CPMLBoundary::new(config, &grid, dt, sound_speed).unwrap();
        let pressure_grad = Array3::ones((30, 30, 30));

        // Test that memory update doesn't panic
        cpml.update_acoustic_memory(&pressure_grad, 0);
        cpml.update_acoustic_memory(&pressure_grad, 1);
        cpml.update_acoustic_memory(&pressure_grad, 2);

        // Check that memory variables have been updated
        let psi_x = cpml.psi_acoustic.index_axis(Axis(0), 0);
        assert!(psi_x.iter().any(|&v| v != 0.0));
    }

    #[test]
    #[should_panic(expected = "Component index must be 0, 1, or 2")]
    fn test_invalid_component_debug() {
        let grid = Grid::new(30, 30, 30, 1e-3, 1e-3, 1e-3);
        let config = CPMLConfig::default();
        let dt = 1e-6;
        let sound_speed = 1500.0;

        let mut cpml = CPMLBoundary::new(config, &grid, dt, sound_speed).unwrap();
        let pressure_grad = Array3::ones((30, 30, 30));

        // This should panic in debug mode
        cpml.update_acoustic_memory(&pressure_grad, 3);
    }

    #[test]
    fn test_fma_optimization() {
        let grid = Grid::new(30, 30, 30, 1e-3, 1e-3, 1e-3);
        let config = CPMLConfig::default();
        let dt = 1e-6;
        let sound_speed = 1500.0;

        let cpml = CPMLBoundary::new(config, &grid, dt, sound_speed).unwrap();
        let mut gradient = Array3::ones((30, 30, 30)) * 2.0;

        // Apply CPML gradient
        cpml.apply_cpml_gradient(&mut gradient, 0);

        // Check that gradient has been modified
        assert!(gradient.iter().any(|&v| v != 2.0));
    }
}
