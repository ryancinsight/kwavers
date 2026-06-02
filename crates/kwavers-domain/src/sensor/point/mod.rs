//! Point Sensors for Arbitrary Position Sampling
//!
//! Implements point sensors that sample acoustic fields at arbitrary
//! (x, y, z) locations using trilinear interpolation from surrounding grid points.
//!
//! # Mathematical Specification
//!
//! Point sensors evaluate the field at locations that may not align with the
//! computational grid using trilinear interpolation:
//!
//! ```text
//! p(x, y, z) = Σᵢⱼₖ wᵢⱼₖ · p[i,j,k]                    (1)
//!
//! where wᵢⱼₖ are trilinear interpolation weights:
//!
//! wᵢⱼₖ = (1 - ξ)(1 - η)(1 - ζ)  for (i,j,k)          (2a)
//!      +    ξ   (1 - η)(1 - ζ)  for (i+1,j,k)        (2b)
//!      + (1 - ξ)   η   (1 - ζ)  for (i,j+1,k)        (2c)
//!      +    ξ      η   (1 - ζ)  for (i+1,j+1,k)      (2d)
//!      + (1 - ξ)(1 - η)   ζ     for (i,j,k+1)        (2e)
//!      +    ξ   (1 - η)   ζ     for (i+1,j,k+1)      (2f)
//!      + (1 - ξ)   η      ζ     for (i,j+1,k+1)      (2g)
//!      +    ξ      η      ζ     for (i+1,j+1,k+1)    (2h)
//!
//! and (ξ, η, ζ) are local coordinates in [0,1]:
//!
//! ξ = (x - xᵢ) / dx                                   (3a)
//! η = (y - yⱼ) / dy                                   (3b)
//! ζ = (z - zₖ) / dz                                   (3c)
//! ```
//!
//! # References
//!
//! 1. Treeby & Cox (2010). "k-Wave: MATLAB toolbox for the simulation and
//!    reconstruction of photoacoustic wave fields." *J. Biomed. Opt.*, 15(2), 021314.
//! 2. Press et al. (2007). *Numerical Recipes* (3rd ed.), Ch. 3: Interpolation.

mod recording;
#[cfg(test)]
mod tests;

use kwavers_core::error::{KwaversError, KwaversResult};
use crate::grid::Grid;
use ndarray::{Array1, Array2, ArrayView3};
use serde::{Deserialize, Serialize};

/// Point sensor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointSensorConfig {
    /// Sensor locations in physical coordinates (m)
    pub locations: Vec<[f64; 3]>,
}

impl PointSensorConfig {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(locations: Vec<[f64; 3]>) -> Self {
        Self { locations }
    }
    /// Validate.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub(super) fn validate(&self, grid: &Grid) -> KwaversResult<()> {
        if self.locations.is_empty() {
            return Err(KwaversError::Validation(
                kwavers_core::error::validation::ValidationError::ConstraintViolation {
                    message: "At least one sensor location required".to_owned(),
                },
            ));
        }

        let (nx, ny, nz) = grid.dimensions();
        let (dx, dy, dz) = grid.spacing();
        let domain_size = [nx as f64 * dx, ny as f64 * dy, nz as f64 * dz];

        for (idx, &[x, y, z]) in self.locations.iter().enumerate() {
            if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                return Err(KwaversError::Validation(
                    kwavers_core::error::validation::ValidationError::ConstraintViolation {
                        message: format!(
                            "Sensor {} location must be finite: [{}, {}, {}]",
                            idx, x, y, z
                        ),
                    },
                ));
            }

            if x < 0.0
                || x > domain_size[0]
                || y < 0.0
                || y > domain_size[1]
                || z < 0.0
                || z > domain_size[2]
            {
                return Err(KwaversError::Validation(
                    kwavers_core::error::validation::ValidationError::ConstraintViolation {
                        message: format!(
                            "Sensor {} location [{}, {}, {}] outside domain [0, 0, 0] to [{}, {}, {}]",
                            idx, x, y, z, domain_size[0], domain_size[1], domain_size[2]
                        ),
                    },
                ));
            }
        }

        Ok(())
    }
}

/// Grid indices and interpolation weights for a point sensor.
///
/// Stores the lower corner indices (i, j, k) and local coordinates (ξ, η, ζ) ∈ [0,1].
#[derive(Debug, Clone)]
pub(super) struct InterpolationData {
    pub(super) indices: [usize; 3],
    pub(super) local_coords: [f64; 3],
}

impl InterpolationData {
    /// Compute trilinear interpolation weights.
    ///
    /// ```text
    /// w[0] = (1-ξ)(1-η)(1-ζ)   [i,   j,   k  ]
    /// w[7] =    ξ    η    ζ    [i+1, j+1, k+1]
    /// ```
    pub(super) fn weights(&self) -> [f64; 8] {
        let [xi, eta, zeta] = self.local_coords;
        let one_minus_xi = 1.0 - xi;
        let one_minus_eta = 1.0 - eta;
        let one_minus_zeta = 1.0 - zeta;

        [
            one_minus_xi * one_minus_eta * one_minus_zeta,
            xi * one_minus_eta * one_minus_zeta,
            one_minus_xi * eta * one_minus_zeta,
            xi * eta * one_minus_zeta,
            one_minus_xi * one_minus_eta * zeta,
            xi * one_minus_eta * zeta,
            one_minus_xi * eta * zeta,
            xi * eta * zeta,
        ]
    }

    pub(super) fn interpolate(&self, field: ArrayView3<f64>) -> f64 {
        let [i, j, k] = self.indices;
        let weights = self.weights();
        let shape = field.shape();

        let i1 = (i + 1).min(shape[0] - 1);
        let j1 = (j + 1).min(shape[1] - 1);
        let k1 = (k + 1).min(shape[2] - 1);

        weights[7].mul_add(
            field[[i1, j1, k1]],
            weights[6].mul_add(
                field[[i, j1, k1]],
                weights[5].mul_add(
                    field[[i1, j, k1]],
                    weights[4].mul_add(
                        field[[i, j, k1]],
                        weights[3].mul_add(
                            field[[i1, j1, k]],
                            weights[2].mul_add(
                                field[[i, j1, k]],
                                weights[0]
                                    .mul_add(field[[i, j, k]], weights[1] * field[[i1, j, k]]),
                            ),
                        ),
                    ),
                ),
            ),
        )
    }
}

/// Point sensor for sampling acoustic fields at arbitrary locations.
///
/// Interpolation data is precomputed during initialization; recording requires
/// only 8 reads + 15 arithmetic ops per sensor per timestep.
#[derive(Debug, Clone)]
pub struct PointSensor {
    pub(super) locations: Vec<[f64; 3]>,
    pub(super) interp_data: Vec<InterpolationData>,
    pub(super) time_history: Vec<Vec<f64>>,
    pub(super) n_timesteps: usize,
}

impl PointSensor {
    /// Create new point sensor with precomputed interpolation data.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: PointSensorConfig, grid: &Grid) -> KwaversResult<Self> {
        config.validate(grid)?;

        let n_sensors = config.locations.len();
        let (dx, dy, dz) = grid.spacing();

        let interp_data: Vec<InterpolationData> = config
            .locations
            .iter()
            .map(|&[x, y, z]| {
                let fi = x / dx;
                let fj = y / dy;
                let fk = z / dz;

                let i = fi.floor() as usize;
                let j = fj.floor() as usize;
                let k = fk.floor() as usize;

                let xi = fi - (i as f64);
                let eta = fj - (j as f64);
                let zeta = fk - (k as f64);

                InterpolationData {
                    indices: [i, j, k],
                    local_coords: [xi, eta, zeta],
                }
            })
            .collect();

        let time_history = vec![Vec::new(); n_sensors];

        Ok(Self {
            locations: config.locations,
            interp_data,
            time_history,
            n_timesteps: 0,
        })
    }

    /// Get time history for specific sensor as 1D array.
    #[must_use]
    pub fn time_history(&self, sensor_idx: usize) -> Option<Array1<f64>> {
        self.time_history
            .get(sensor_idx)
            .map(|history| Array1::from_vec(history.clone()))
    }

    /// Get all time histories as 2D array [n_sensors × n_timesteps].
    #[must_use]
    pub fn all_time_histories(&self) -> Array2<f64> {
        let n_sensors = self.locations.len();
        let n_timesteps = self.n_timesteps;

        let mut histories = Array2::<f64>::zeros((n_sensors, n_timesteps));
        for (i, history) in self.time_history.iter().enumerate() {
            for (j, &value) in history.iter().enumerate() {
                histories[[i, j]] = value;
            }
        }
        histories
    }

    #[must_use]
    pub fn locations(&self) -> &[[f64; 3]] {
        &self.locations
    }

    #[must_use]
    pub fn n_sensors(&self) -> usize {
        self.locations.len()
    }

    #[must_use]
    pub fn n_timesteps(&self) -> usize {
        self.n_timesteps
    }

    pub fn clear(&mut self) {
        for history in &mut self.time_history {
            history.clear();
        }
        self.n_timesteps = 0;
    }
}
