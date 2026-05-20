use crate::core::error::{ConfigError, KwaversResult};
use ndarray::ArrayViewMut3;
use serde::{Deserialize, Serialize};

mod boundary_impl;

// Exponential scaling factor for PML absorption profile — Berenger (1994)
const PML_EXPONENTIAL_SCALING_FACTOR: f64 = 0.1;

/// Perfectly Matched Layer (PML) boundary condition for absorbing outgoing waves.
#[derive(Debug, Clone)]
pub struct DomainPMLBoundary {
    pub(super) acoustic_damping_x: Vec<f64>,
    pub(super) acoustic_damping_y: Vec<f64>,
    pub(super) acoustic_damping_z: Vec<f64>,
    pub(super) light_damping_x: Vec<f64>,
    pub(super) light_damping_y: Vec<f64>,
    pub(super) light_damping_z: Vec<f64>,
    pub(super) thickness: usize,
}

/// Configuration for PML boundary layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainPmlConfig {
    pub thickness: usize,
    pub sigma_max_acoustic: f64,
    pub sigma_max_light: f64,
    pub alpha_max_acoustic: f64,
    pub alpha_max_light: f64,
    pub kappa_max_acoustic: f64,
    pub kappa_max_light: f64,
    pub target_reflection: Option<f64>,
}

impl Default for DomainPmlConfig {
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

impl DomainPmlConfig {
    /// With thickness.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn with_thickness(mut self, thickness: usize) -> Self {
        self.thickness = thickness;
        self
    }
    /// With reflection coefficient.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn with_reflection_coefficient(mut self, reflection: f64) -> Self {
        self.target_reflection = Some(reflection);
        self
    }
    /// Validate.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self.thickness == 0 {
            return Err(ConfigError::InvalidValue {
                parameter: "thickness".to_owned(),
                value: self.thickness.to_string(),
                constraint: "PML thickness must be > 0".to_owned(),
            }
            .into());
        }

        if self.sigma_max_acoustic < 0.0 || self.sigma_max_light < 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "sigma_max".to_owned(),
                value: format!(
                    "acoustic: {}, light: {}",
                    self.sigma_max_acoustic, self.sigma_max_light
                ),
                constraint: "Sigma values must be >= 0".to_owned(),
            }
            .into());
        }

        Ok(())
    }
}

impl DomainPMLBoundary {
    /// New.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: DomainPmlConfig) -> KwaversResult<Self> {
        config.validate()?;

        let acoustic_profile =
            Self::damping_profile(config.thickness, 100, 1.0, config.sigma_max_acoustic, 2);
        let light_profile =
            Self::damping_profile(config.thickness, 100, 1.0, config.sigma_max_light, 2);

        Ok(Self {
            acoustic_damping_x: acoustic_profile.clone(),
            acoustic_damping_y: acoustic_profile.clone(),
            acoustic_damping_z: acoustic_profile,
            light_damping_x: light_profile.clone(),
            light_damping_y: light_profile.clone(),
            light_damping_z: light_profile,
            thickness: config.thickness,
        })
    }
    /// With defaults.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn with_defaults() -> KwaversResult<Self> {
        Self::new(DomainPmlConfig::default())
    }

    fn damping_profile(
        thickness: usize,
        _length: usize,
        dx: f64,
        sigma_max: f64,
        order: usize,
    ) -> Vec<f64> {
        let mut profile = vec![0.0; thickness];

        let target_reflection: f64 = 1e-6;
        let reference_sigma =
            -((order + 1) as f64) * target_reflection.ln() / (2.0 * thickness as f64 * dx);
        let sigma_eff = sigma_max.min(reference_sigma * 2.0);

        for (i, profile_val) in profile.iter_mut().enumerate() {
            let dist_from_boundary = i as f64;
            let normalized_dist = (thickness as f64 - dist_from_boundary) / thickness as f64;

            let polynomial_factor = normalized_dist.powi(order as i32);
            let exponential_factor = (-2.0 * normalized_dist).exp();

            *profile_val = sigma_eff
                * polynomial_factor
                * PML_EXPONENTIAL_SCALING_FACTOR.mul_add(exponential_factor, 1.0);
        }

        profile
    }

    #[inline]
    pub(super) fn apply_damping(val: &mut f64, damping: f64) {
        // Profile values are dimensionless per-cell attenuation exponents; apply directly.
        if damping > 0.0 {
            *val *= (-damping).exp();
        }
    }

    #[inline]
    pub(super) fn combine_damping(d_x: f64, d_y: f64, d_z: f64) -> f64 {
        d_x.max(d_y).max(d_z)
    }

    #[inline]
    pub(super) fn get_damping(&self, idx: usize, profile: &[f64], max_dim: usize) -> f64 {
        if idx < self.thickness {
            profile[idx]
        } else if idx >= max_dim - self.thickness {
            let dist = max_dim - 1 - idx;
            profile[dist]
        } else {
            0.0
        }
    }

    #[inline]
    pub(super) fn precompute_exp_factors(profile: &[f64]) -> Vec<f64> {
        // Profile values are dimensionless per-cell attenuation exponents (computed with dx=1.0
        // in damping_profile); apply directly as exp(-sigma) without grid-spacing scaling.
        profile
            .iter()
            .map(|&d| if d > 0.0 { (-d).exp() } else { 1.0 })
            .collect()
    }

    #[inline]
    pub(super) fn precompute_full_exp_factors(
        exp_profile: &[f64],
        dim_size: usize,
        thickness: usize,
    ) -> Vec<f64> {
        let mut factors = vec![1.0; dim_size];
        for i in 0..thickness {
            factors[i] = exp_profile[i];
            factors[dim_size - 1 - i] = exp_profile[i];
        }
        factors
    }

    /// Apply acoustic PML with custom damping factor
    pub fn apply_acoustic_with_factor(
        &mut self,
        mut field: ArrayViewMut3<f64>,
        grid: &crate::domain::grid::Grid,
        time_step: usize,
        damping_factor: f64,
    ) {
        log::trace!(
            "Applying acoustic PML with factor {} at step {}",
            damping_factor,
            time_step
        );
        let (nx, ny, nz) = grid.dimensions();
        let t = self.thickness;

        for i in 0..t {
            let d_x = self.acoustic_damping_x[i];
            for j in 0..ny {
                for k in 0..nz {
                    let d_y = self.get_damping(j, &self.acoustic_damping_y, ny);
                    let d_z = self.get_damping(k, &self.acoustic_damping_z, nz);
                    Self::apply_damping(
                        &mut field[[i, j, k]],
                        Self::combine_damping(d_x, d_y, d_z) * damping_factor,
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
                    );
                }
            }
        }

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
                        );
                    }
                }
            }

            let y_start = t;
            let y_end = ny - t;
            if y_end > y_start {
                for k in 0..t {
                    let d_z = self.acoustic_damping_z[k];
                    for i in x_start..x_end {
                        for j in y_start..y_end {
                            Self::apply_damping(&mut field[[i, j, k]], d_z * damping_factor);
                        }
                    }
                    let rk = nz - 1 - k;
                    let d_z_r = self.acoustic_damping_z[k];
                    for i in x_start..x_end {
                        for j in y_start..y_end {
                            Self::apply_damping(&mut field[[i, j, rk]], d_z_r * damping_factor);
                        }
                    }
                }
            }
        }
    }
}
