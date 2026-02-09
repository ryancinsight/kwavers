use crate::core::error::{ConfigError, KwaversError, KwaversResult, ValidationError};
use crate::domain::boundary::Boundary;
use crate::domain::grid::Grid;
use log::trace;
use ndarray::{Array3, ArrayViewMut3};

use rustfft::num_complex::Complex;

use serde::{Deserialize, Serialize};

// Physical constants for PML boundary parameters
/// Exponential scaling factor for PML absorption profile
/// This factor adds a small exponential component to the polynomial PML profile
/// to modify absorption characteristics at grazing angles. The value 0.1 provides
/// a 10% scaling without destabilizing the absorption profile.
/// Based on: Berenger, "A perfectly matched layer for absorption of electromagnetic waves"
const PML_EXPONENTIAL_SCALING_FACTOR: f64 = 0.1;

/// Perfectly Matched Layer (PML) boundary condition for absorbing outgoing waves.
///
/// This implementation uses a polynomial grading of the absorption profile
/// with optional backing by a theoretical model for automatic parameter selection.
#[derive(Debug, Clone)]
pub struct PMLBoundary {
    /// Pre-computed damping profiles for each dimension
    acoustic_damping_x: Vec<f64>,
    acoustic_damping_y: Vec<f64>,
    acoustic_damping_z: Vec<f64>,
    light_damping_x: Vec<f64>,
    light_damping_y: Vec<f64>,
    light_damping_z: Vec<f64>,
    thickness: usize,
}

/// Configuration for PML boundary layer
/// Follows SOLID principles by grouping related parameters together
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PMLConfig {
    pub thickness: usize,
    pub sigma_max_acoustic: f64,
    pub sigma_max_light: f64,
    pub alpha_max_acoustic: f64,
    pub alpha_max_light: f64,
    pub kappa_max_acoustic: f64,
    pub kappa_max_light: f64,
    pub target_reflection: Option<f64>,
}

impl Default for PMLConfig {
    fn default() -> Self {
        Self {
            thickness: 10,
            sigma_max_acoustic: 2.0,
            sigma_max_light: 1.0,
            alpha_max_acoustic: 0.0,
            alpha_max_light: 0.0,
            kappa_max_acoustic: 1.0,
            kappa_max_light: 1.0,
            target_reflection: Some(1e-4),
        }
    }
}

impl PMLConfig {
    /// Set PML thickness
    #[must_use]
    pub fn with_thickness(mut self, thickness: usize) -> Self {
        self.thickness = thickness;
        self
    }

    /// Set reflection coefficient
    #[must_use]
    pub fn with_reflection_coefficient(mut self, reflection: f64) -> Self {
        self.target_reflection = Some(reflection);
        self
    }

    /// Validate PML configuration parameters
    /// Follows SOLID Single Responsibility Principle
    pub fn validate(&self) -> KwaversResult<()> {
        if self.thickness == 0 {
            return Err(ConfigError::InvalidValue {
                parameter: "thickness".to_string(),
                value: self.thickness.to_string(),
                constraint: "PML thickness must be > 0".to_string(),
            }
            .into());
        }

        if self.sigma_max_acoustic < 0.0 || self.sigma_max_light < 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "sigma_max".to_string(),
                value: format!(
                    "acoustic: {}, light: {}",
                    self.sigma_max_acoustic, self.sigma_max_light
                ),
                constraint: "Sigma values must be >= 0".to_string(),
            }
            .into());
        }

        Ok(())
    }
}

impl PMLBoundary {
    /// Create new PML boundary with configuration struct
    /// Follows SOLID principles by reducing parameter coupling
    pub fn new(config: PMLConfig) -> KwaversResult<Self> {
        config.validate()?;

        // Create damping profiles based on configuration
        let acoustic_profile =
            Self::damping_profile(config.thickness, 100, 1.0, config.sigma_max_acoustic, 2);
        let light_profile =
            Self::damping_profile(config.thickness, 100, 1.0, config.sigma_max_light, 2);

        Ok(Self {
            acoustic_damping_x: acoustic_profile.clone(),
            acoustic_damping_y: acoustic_profile.clone(),
            acoustic_damping_z: acoustic_profile.clone(),
            light_damping_x: light_profile.clone(),
            light_damping_y: light_profile.clone(),
            light_damping_z: light_profile,
            thickness: config.thickness,
        })
    }

    /// Create with default configuration
    pub fn with_defaults() -> KwaversResult<Self> {
        Self::new(PMLConfig::default())
    }

    /// Creates a damping profile for a PML layer with frequency-dependent absorption.
    ///
    /// # Arguments
    ///
    /// * `thickness` - PML thickness in grid points
    /// * `length` - Total domain length in grid points
    /// * `dx` - Grid spacing
    /// * `sigma_max` - Maximum absorption coefficient
    /// * `order` - Polynomial order for profile grading
    fn damping_profile(
        thickness: usize,
        _length: usize, // Unused in local profile generation
        dx: f64,
        sigma_max: f64,
        order: usize,
    ) -> Vec<f64> {
        // We only store the profile for the thickness itself, symmetrical
        // Profile is stored as [sigma(0), sigma(1), ... sigma(thickness-1)]
        // where sigma(0) is at the deepest point of PML (boundary) and sigma(thickness-1) is interface
        let mut profile = vec![0.0; thickness];

        // PML profile with exponential absorption characteristics
        // Theoretical reference sigma for reflection coefficient R
        let target_reflection: f64 = 1e-6; // -120 dB reflection
        let reference_sigma =
            -((order + 1) as f64) * target_reflection.ln() / (2.0 * thickness as f64 * dx);
        let sigma_eff = sigma_max.min(reference_sigma * 2.0); // Don't exceed theoretical reference

        for (i, profile_val) in profile.iter_mut().enumerate() {
            // Distance from interface (normalized): 0 at interface, 1 at boundary
            // In our loop i=0 is boundary, i=thickness-1 is interface
            // So distance is (thickness - 1 - i) / thickness
            // Wait, legacy implementation used full length. Let's standardize.
            // Let profile[d] be damping at distance d nodes from boundary.
            // d=0 is the boundary node.

            let dist_from_boundary = i as f64;
            let normalized_dist = (thickness as f64 - dist_from_boundary) / thickness as f64;
            // normalized_dist is 1 at boundary (i=0), 0 at interface

            // Standard polynomial grading: sigma(x) = sigma_max * (x/L)^n
            let polynomial_factor = normalized_dist.powi(order as i32);

            // Add exponential component for grazing angle absorption
            let exponential_factor = (-2.0 * normalized_dist).exp();

            *profile_val = sigma_eff
                * polynomial_factor
                * (1.0 + PML_EXPONENTIAL_SCALING_FACTOR * exponential_factor);
        }

        profile
    }

    /// Applies a pre-computed damping factor to a field value
    #[inline]
    fn apply_damping(val: &mut f64, damping: f64, dx: f64) {
        if damping > 0.0 {
            *val *= (-damping * dx).exp();
        }
    }

    #[inline]
    fn combine_damping(d_x: f64, d_y: f64, d_z: f64) -> f64 {
        d_x.max(d_y).max(d_z)
    }

    /// Get damping factor at a specific index
    #[inline]
    fn get_damping(&self, idx: usize, profile: &[f64], max_dim: usize) -> f64 {
        if idx < self.thickness {
            profile[idx]
        } else if idx >= max_dim - self.thickness {
            let dist = max_dim - 1 - idx;
            profile[dist]
        } else {
            0.0
        }
    }

    /// Precompute exponential decay factors for a damping profile.
    /// Returns `exp(-damping[i] * dx)` for each profile entry, avoiding per-point exp() calls.
    #[inline]
    fn precompute_exp_factors(profile: &[f64], dx: f64) -> Vec<f64> {
        profile
            .iter()
            .map(|&d| if d > 0.0 { (-d * dx).exp() } else { 1.0 })
            .collect()
    }

    /// Precompute full-dimension exp factor array for a given axis.
    /// Interior points get factor 1.0 (no damping), boundary points get precomputed decay.
    #[inline]
    fn precompute_full_exp_factors(exp_profile: &[f64], dim_size: usize, thickness: usize) -> Vec<f64> {
        let mut factors = vec![1.0; dim_size];
        for i in 0..thickness {
            factors[i] = exp_profile[i];
            factors[dim_size - 1 - i] = exp_profile[i];
        }
        factors
    }
}

impl Boundary for PMLBoundary {
    fn apply_acoustic(
        &mut self,
        mut field: ArrayViewMut3<f64>,
        grid: &Grid,
        time_step: usize,
    ) -> crate::core::error::KwaversResult<()> {
        trace!("Applying spatial acoustic PML at step {}", time_step);
        let dx = grid.dx;
        let (nx, ny, nz) = grid.dimensions();
        let t = self.thickness;

        if 2 * t >= nx {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "PML thickness {} incompatible with grid nx={}; require 2*thickness < nx",
                        t, nx
                    ),
                },
            ));
        }
        if ny > 1 && 2 * t >= ny {
            return Err(KwaversError::Validation(ValidationError::ConstraintViolation {
                message: format!(
                    "PML thickness {} incompatible with grid ny={}; require 2*thickness < ny for y-PML",
                    t, ny
                ),
            }));
        }
        if nz > 1 && 2 * t >= nz {
            return Err(KwaversError::Validation(ValidationError::ConstraintViolation {
                message: format!(
                    "PML thickness {} incompatible with grid nz={}; require 2*thickness < nz for z-PML",
                    t, nz
                ),
            }));
        }

        let apply_y = ny > 1;
        let apply_z = nz > 1;

        // Precompute all exp decay factors once — O(thickness) exp() calls instead of O(N²)
        let exp_x = Self::precompute_exp_factors(&self.acoustic_damping_x, dx);
        let exp_y_full = if apply_y {
            Self::precompute_full_exp_factors(
                &Self::precompute_exp_factors(&self.acoustic_damping_y, dx),
                ny,
                t,
            )
        } else {
            vec![1.0; ny]
        };
        let exp_z_full = if apply_z {
            Self::precompute_full_exp_factors(
                &Self::precompute_exp_factors(&self.acoustic_damping_z, dx),
                nz,
                t,
            )
        } else {
            vec![1.0; nz]
        };

        // Apply X boundaries (full Y-Z plane, includes corners)
        for i in 0..t {
            let fx = exp_x[i];
            // Left boundary
            for j in 0..ny {
                let fy = exp_y_full[j];
                for k in 0..nz {
                    let fz = exp_z_full[k];
                    // combine_damping uses max → exp(-max*dx) = min of individual factors
                    field[[i, j, k]] *= fx.min(fy).min(fz);
                }
            }
            // Right boundary
            let ri = nx - 1 - i;
            for j in 0..ny {
                let fy = exp_y_full[j];
                for k in 0..nz {
                    let fz = exp_z_full[k];
                    field[[ri, j, k]] *= fx.min(fy).min(fz);
                }
            }
        }

        // Apply Y boundaries (excluding X corners already handled)
        let x_start = t;
        let x_end = nx - t;

        if apply_y && x_end > x_start {
            let exp_y = Self::precompute_exp_factors(&self.acoustic_damping_y, dx);
            for j in 0..t {
                let fy = exp_y[j];
                // Bottom boundary
                for i in x_start..x_end {
                    for k in 0..nz {
                        let fz = exp_z_full[k];
                        field[[i, j, k]] *= fy.min(fz);
                    }
                }
                // Top boundary
                let rj = ny - 1 - j;
                for i in x_start..x_end {
                    for k in 0..nz {
                        let fz = exp_z_full[k];
                        field[[i, rj, k]] *= fy.min(fz);
                    }
                }
            }

            let y_start = t;
            let y_end = ny - t;

            // Apply Z boundaries (excluding X and Y corners)
            if apply_z && y_end > y_start {
                let exp_z = Self::precompute_exp_factors(&self.acoustic_damping_z, dx);
                for k in 0..t {
                    let fz = exp_z[k];
                    // Front boundary
                    for i in x_start..x_end {
                        for j in y_start..y_end {
                            field[[i, j, k]] *= fz;
                        }
                    }
                    // Back boundary
                    let rk = nz - 1 - k;
                    for i in x_start..x_end {
                        for j in y_start..y_end {
                            field[[i, j, rk]] *= fz;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn apply_acoustic_freq(
        &mut self,
        field: &mut Array3<Complex<f64>>,
        grid: &Grid,
        time_step: usize,
    ) -> crate::core::error::KwaversResult<()> {
        trace!(
            "Applying frequency domain acoustic PML at step {}",
            time_step
        );
        let dx = grid.dx;
        let (nx, ny, nz) = grid.dimensions();
        let t = self.thickness;

        if 2 * t >= nx {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "PML thickness {} incompatible with grid nx={}; require 2*thickness < nx",
                        t, nx
                    ),
                },
            ));
        }
        if ny > 1 && 2 * t >= ny {
            return Err(KwaversError::Validation(ValidationError::ConstraintViolation {
                message: format!(
                    "PML thickness {} incompatible with grid ny={}; require 2*thickness < ny for y-PML",
                    t, ny
                ),
            }));
        }
        if nz > 1 && 2 * t >= nz {
            return Err(KwaversError::Validation(ValidationError::ConstraintViolation {
                message: format!(
                    "PML thickness {} incompatible with grid nz={}; require 2*thickness < nz for z-PML",
                    t, nz
                ),
            }));
        }

        let apply_y = ny > 1;
        let apply_z = nz > 1;

        // Precompute exp decay factors once
        let exp_x = Self::precompute_exp_factors(&self.acoustic_damping_x, dx);
        let exp_y_full = if apply_y {
            Self::precompute_full_exp_factors(
                &Self::precompute_exp_factors(&self.acoustic_damping_y, dx),
                ny,
                t,
            )
        } else {
            vec![1.0; ny]
        };
        let exp_z_full = if apply_z {
            Self::precompute_full_exp_factors(
                &Self::precompute_exp_factors(&self.acoustic_damping_z, dx),
                nz,
                t,
            )
        } else {
            vec![1.0; nz]
        };

        // Apply X boundaries (full Y-Z plane) - Handles corners fully
        for i in 0..t {
            let fx = exp_x[i];
            for j in 0..ny {
                let fy = exp_y_full[j];
                for k in 0..nz {
                    let fz = exp_z_full[k];
                    let decay = fx.min(fy).min(fz);
                    field[[i, j, k]].re *= decay;
                    field[[i, j, k]].im *= decay;
                }
            }
            let ri = nx - 1 - i;
            for j in 0..ny {
                let fy = exp_y_full[j];
                for k in 0..nz {
                    let fz = exp_z_full[k];
                    let decay = fx.min(fy).min(fz);
                    field[[ri, j, k]].re *= decay;
                    field[[ri, j, k]].im *= decay;
                }
            }
        }

        // Apply Y boundaries (excluding X regions)
        let x_start = t;
        let x_end = nx - t;

        if apply_y && x_end > x_start {
            let exp_y = Self::precompute_exp_factors(&self.acoustic_damping_y, dx);
            for j in 0..t {
                let fy = exp_y[j];
                for i in x_start..x_end {
                    for k in 0..nz {
                        let fz = exp_z_full[k];
                        let decay = fy.min(fz);
                        field[[i, j, k]].re *= decay;
                        field[[i, j, k]].im *= decay;
                    }
                }
                let rj = ny - 1 - j;
                for i in x_start..x_end {
                    for k in 0..nz {
                        let fz = exp_z_full[k];
                        let decay = fy.min(fz);
                        field[[i, rj, k]].re *= decay;
                        field[[i, rj, k]].im *= decay;
                    }
                }
            }

            // Apply Z boundaries (excluding X and Y regions)
            let y_start = t;
            let y_end = ny - t;

            if apply_z && y_end > y_start {
                let exp_z = Self::precompute_exp_factors(&self.acoustic_damping_z, dx);
                for k in 0..t {
                    let fz = exp_z[k];
                    for i in x_start..x_end {
                        for j in y_start..y_end {
                            field[[i, j, k]].re *= fz;
                            field[[i, j, k]].im *= fz;
                        }
                    }
                    let rk = nz - 1 - k;
                    for i in x_start..x_end {
                        for j in y_start..y_end {
                            field[[i, j, rk]].re *= fz;
                            field[[i, j, rk]].im *= fz;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn apply_light(&mut self, mut field: ArrayViewMut3<f64>, grid: &Grid, time_step: usize) {
        trace!("Applying light PML at step {}", time_step);
        let dx = grid.dx;
        let (nx, ny, nz) = grid.dimensions();
        let t = self.thickness;
        let apply_y = ny > 1 && 2 * t < ny;
        let apply_z = nz > 1 && 2 * t < nz;

        // X boundaries
        for i in 0..t.min(nx.saturating_sub(1)) {
            let d_x = self.light_damping_x[i];
            for j in 0..ny {
                for k in 0..nz {
                    let d_y = if apply_y {
                        self.get_damping(j, &self.light_damping_y, ny)
                    } else {
                        0.0
                    };
                    let d_z = if apply_z {
                        self.get_damping(k, &self.light_damping_z, nz)
                    } else {
                        0.0
                    };
                    Self::apply_damping(&mut field[[i, j, k]], d_x + d_y + d_z, dx);
                }
            }
            let ri = nx - 1 - i;
            let d_x_r = self.light_damping_x[i];
            for j in 0..ny {
                for k in 0..nz {
                    let d_y = if apply_y {
                        self.get_damping(j, &self.light_damping_y, ny)
                    } else {
                        0.0
                    };
                    let d_z = if apply_z {
                        self.get_damping(k, &self.light_damping_z, nz)
                    } else {
                        0.0
                    };
                    Self::apply_damping(&mut field[[ri, j, k]], d_x_r + d_y + d_z, dx);
                }
            }
        }

        // Y boundaries
        let x_start = t;
        let x_end = nx - t;
        if x_end > x_start {
            for j in 0..t {
                let d_y = self.light_damping_y[j];
                for i in x_start..x_end {
                    for k in 0..nz {
                        let d_z = self.get_damping(k, &self.light_damping_z, nz);
                        Self::apply_damping(&mut field[[i, j, k]], d_y + d_z, dx);
                    }
                }
                let rj = ny - 1 - j;
                let d_y_r = self.light_damping_y[j];
                for i in x_start..x_end {
                    for k in 0..nz {
                        let d_z = self.get_damping(k, &self.light_damping_z, nz);
                        Self::apply_damping(&mut field[[i, rj, k]], d_y_r + d_z, dx);
                    }
                }
            }

            // Z boundaries
            let y_start = t;
            let y_end = ny - t;
            if y_end > y_start {
                for k in 0..t {
                    let d_z = self.light_damping_z[k];
                    for i in x_start..x_end {
                        for j in y_start..y_end {
                            Self::apply_damping(&mut field[[i, j, k]], d_z, dx);
                        }
                    }
                    let rk = nz - 1 - k;
                    let d_z_r = self.light_damping_z[k];
                    for i in x_start..x_end {
                        for j in y_start..y_end {
                            Self::apply_damping(&mut field[[i, j, rk]], d_z_r, dx);
                        }
                    }
                }
            }
        }
    }
}

impl PMLBoundary {
    /// Apply acoustic PML with custom damping factor
    /// Follows Open/Closed Principle: Extends functionality without modifying existing code
    pub fn apply_acoustic_with_factor(
        &mut self,
        mut field: ArrayViewMut3<f64>,
        grid: &Grid,
        time_step: usize,
        damping_factor: f64,
    ) {
        trace!(
            "Applying acoustic PML with factor {} at step {}",
            damping_factor,
            time_step
        );
        let dx = grid.dx;
        let (nx, ny, nz) = grid.dimensions();
        let t = self.thickness;

        // Apply X boundaries
        for i in 0..t {
            let d_x = self.acoustic_damping_x[i];
            for j in 0..ny {
                for k in 0..nz {
                    let d_y = self.get_damping(j, &self.acoustic_damping_y, ny);
                    let d_z = self.get_damping(k, &self.acoustic_damping_z, nz);
                    Self::apply_damping(
                        &mut field[[i, j, k]],
                        Self::combine_damping(d_x, d_y, d_z) * damping_factor,
                        dx,
                    );
                }
            }
            let ri = nx - 1 - i;
            let d_x_r = self.acoustic_damping_x[i];
            for j in 0..ny {
                for k in 0..nz {
                    let d_y = self.get_damping(j, &self.acoustic_damping_y, ny);
                    let d_z = self.get_damping(k, &self.acoustic_damping_z, nz);
                    Self::apply_damping(
                        &mut field[[ri, j, k]],
                        Self::combine_damping(d_x_r, d_y, d_z) * damping_factor,
                        dx,
                    );
                }
            }
        }

        // Apply Y boundaries
        let x_start = t;
        let x_end = nx - t;
        if x_end > x_start {
            for j in 0..t {
                let d_y = self.acoustic_damping_y[j];
                for i in x_start..x_end {
                    for k in 0..nz {
                        let d_z = self.get_damping(k, &self.acoustic_damping_z, nz);
                        Self::apply_damping(
                            &mut field[[i, j, k]],
                            Self::combine_damping(0.0, d_y, d_z) * damping_factor,
                            dx,
                        );
                    }
                }
                let rj = ny - 1 - j;
                let d_y_r = self.acoustic_damping_y[j];
                for i in x_start..x_end {
                    for k in 0..nz {
                        let d_z = self.get_damping(k, &self.acoustic_damping_z, nz);
                        Self::apply_damping(
                            &mut field[[i, rj, k]],
                            Self::combine_damping(0.0, d_y_r, d_z) * damping_factor,
                            dx,
                        );
                    }
                }
            }

            // Apply Z boundaries
            let y_start = t;
            let y_end = ny - t;
            if y_end > y_start {
                for k in 0..t {
                    let d_z = self.acoustic_damping_z[k];
                    for i in x_start..x_end {
                        for j in y_start..y_end {
                            Self::apply_damping(&mut field[[i, j, k]], d_z * damping_factor, dx);
                        }
                    }
                    let rk = nz - 1 - k;
                    let d_z_r = self.acoustic_damping_z[k];
                    for i in x_start..x_end {
                        for j in y_start..y_end {
                            Self::apply_damping(&mut field[[i, j, rk]], d_z_r * damping_factor, dx);
                        }
                    }
                }
            }
        }
    }
}
