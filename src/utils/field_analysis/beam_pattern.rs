//! Beam pattern analysis and directivity calculations

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::{Array2, ArrayView3};
use std::f64::consts::PI;

/// Beam pattern analysis configuration
#[derive(Debug, Clone)]
pub struct BeamPatternConfig {
    /// Analysis frequency (Hz)
    pub frequency: f64,
    /// Sound speed (m/s)
    pub sound_speed: f64,
    /// Far-field distance calculation method
    pub far_field_method: FarFieldMethod,
    /// Angular resolution (radians)
    pub angular_resolution: f64,
}

/// Far-field calculation methods
#[derive(Debug, Clone)]
pub enum FarFieldMethod {
    /// Fresnel approximation
    Fresnel,
    /// Fraunhofer approximation
    Fraunhofer,
    /// Full calculation
    Exact,
}

impl Default for BeamPatternConfig {
    fn default() -> Self {
        Self {
            frequency: 1e6, // 1 MHz
            sound_speed: crate::physics::constants::SOUND_SPEED_WATER,
            far_field_method: FarFieldMethod::Fraunhofer,
            angular_resolution: PI / 180.0, // 1 degree
        }
    }
}

/// Calculate beam pattern from pressure field
///
/// # Arguments
/// * `pressure_field` - 3D pressure field
/// * `grid` - Spatial grid
/// * `config` - Beam pattern configuration
///
/// # Returns
/// * 2D array of beam pattern (theta, phi)
pub fn calculate_beam_pattern(
    pressure_field: ArrayView3<f64>,
    grid: &Grid,
    config: &BeamPatternConfig,
) -> KwaversResult<Array2<f64>> {
    let wavelength = config.sound_speed / config.frequency;
    let k = 2.0 * PI / wavelength;

    // Angular sampling
    let n_theta = ((2.0 * PI) / config.angular_resolution) as usize;
    let n_phi = (PI / config.angular_resolution) as usize;

    let mut pattern = Array2::zeros((n_theta, n_phi));

    // Calculate far-field distance
    let _far_field_distance =
        calculate_far_field_distance(grid, wavelength, &config.far_field_method);

    // Calculate beam pattern for each angle
    for i_theta in 0..n_theta {
        let theta = i_theta as f64 * config.angular_resolution;

        for i_phi in 0..n_phi {
            let phi = i_phi as f64 * config.angular_resolution;

            // Direction vector
            let dir_x = phi.sin() * theta.cos();
            let dir_y = phi.sin() * theta.sin();
            let dir_z = phi.cos();

            // Calculate far-field pressure using Rayleigh integral
            let mut pressure_complex = num_complex::Complex::new(0.0, 0.0);

            for ix in 0..grid.nx {
                for iy in 0..grid.ny {
                    for iz in 0..grid.nz {
                        let x = ix as f64 * grid.dx;
                        let y = iy as f64 * grid.dy;
                        let z = iz as f64 * grid.dz;

                        // Phase from this point to far-field
                        let phase = k * (x * dir_x + y * dir_y + z * dir_z);

                        // Add contribution
                        let p = pressure_field[[ix, iy, iz]];
                        pressure_complex += p * num_complex::Complex::from_polar(1.0, -phase);
                    }
                }
            }

            pattern[[i_theta, i_phi]] = pressure_complex.norm();
        }
    }

    // Normalize to maximum
    let max_val = pattern.iter().fold(0.0f64, |a: f64, &b| a.max(b.abs()));
    if max_val > 0.0 {
        pattern /= max_val;
    }

    Ok(pattern)
}

/// Calculate directivity index
#[must_use]
pub fn calculate_directivity(beam_pattern: &Array2<f64>) -> f64 {
    let max_val = beam_pattern.iter().fold(0.0f64, |a: f64, &b| a.max(b.abs()));
    let mean_val = beam_pattern.iter().sum::<f64>() / beam_pattern.len() as f64;

    if mean_val > 0.0 {
        10.0 * (max_val / mean_val).log10()
    } else {
        0.0
    }
}

/// Calculate far-field distance
fn calculate_far_field_distance(grid: &Grid, wavelength: f64, method: &FarFieldMethod) -> f64 {
    let aperture = ((grid.nx as f64 * grid.dx).powi(2) + (grid.ny as f64 * grid.dy).powi(2)).sqrt();

    match method {
        FarFieldMethod::Fresnel => aperture.powi(2) / wavelength,
        FarFieldMethod::Fraunhofer => 2.0 * aperture.powi(2) / wavelength,
        FarFieldMethod::Exact => 10.0 * aperture.powi(2) / wavelength,
    }
}
