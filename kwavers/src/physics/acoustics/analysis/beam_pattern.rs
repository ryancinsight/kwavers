//! Beam pattern analysis and directivity calculations

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
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
            frequency: 1e6,      // 1 MHz
            sound_speed: 1500.0, // Default sound speed
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
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
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
            let mut pressure_complex = crate::math::fft::Complex64::new(0.0, 0.0);

            for ix in 0..grid.nx {
                for iy in 0..grid.ny {
                    for iz in 0..grid.nz {
                        let x = ix as f64 * grid.dx;
                        let y = iy as f64 * grid.dy;
                        let z = iz as f64 * grid.dz;

                        // Phase from this point to far-field
                        let phase = k * z.mul_add(dir_z, x.mul_add(dir_x, y * dir_y));

                        // Add contribution
                        let p = pressure_field[[ix, iy, iz]];
                        pressure_complex +=
                            p * crate::math::fft::Complex64::from_polar(1.0, -phase);
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
    let max_val = beam_pattern
        .iter()
        .fold(0.0f64, |a: f64, &b| a.max(b.abs()));
    let mean_val = beam_pattern.iter().sum::<f64>() / beam_pattern.len() as f64;

    if mean_val > 0.0 {
        10.0 * (max_val / mean_val).log10()
    } else {
        0.0
    }
}

/// Calculate far-field distance
fn calculate_far_field_distance(grid: &Grid, wavelength: f64, method: &FarFieldMethod) -> f64 {
    let aperture = (grid.nx as f64 * grid.dx).hypot(grid.ny as f64 * grid.dy);

    match method {
        FarFieldMethod::Fresnel => aperture.powi(2) / wavelength,
        FarFieldMethod::Fraunhofer => 2.0 * aperture.powi(2) / wavelength,
        FarFieldMethod::Exact => 10.0 * aperture.powi(2) / wavelength,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use ndarray::{Array2, Array3};

    // ── BeamPatternConfig::default ────────────────────────────────────────────

    /// Default config: 1 MHz, 1500 m/s, Fraunhofer method, 1° angular resolution.
    #[test]
    fn beam_pattern_config_default_values_match_documentation() {
        let cfg = BeamPatternConfig::default();
        assert!((cfg.frequency - 1e6).abs() < 1.0, "frequency must be 1 MHz");
        assert!(
            (cfg.sound_speed - 1500.0).abs() < 1e-10,
            "sound_speed must be 1500 m/s"
        );
        assert!(
            (cfg.angular_resolution - PI / 180.0).abs() < 1e-12,
            "resolution must be 1°"
        );
        assert!(matches!(cfg.far_field_method, FarFieldMethod::Fraunhofer));
    }

    // ── calculate_directivity ─────────────────────────────────────────────────

    /// Uniform pattern (all ones) → max=1, mean=1 → DI = 10·log10(1/1) = 0 dB.
    #[test]
    fn calculate_directivity_zero_db_for_uniform_pattern() {
        let pattern = Array2::<f64>::ones((10, 10));
        let di = calculate_directivity(&pattern);
        assert!(
            di.abs() < 1e-10,
            "uniform pattern must give DI=0 dB (got {di:.6})"
        );
    }

    /// Zero pattern (all zeros) → mean=0 → DI = 0.0 (guarded branch).
    #[test]
    fn calculate_directivity_zero_for_zero_pattern() {
        let pattern = Array2::<f64>::zeros((10, 10));
        let di = calculate_directivity(&pattern);
        assert_eq!(di, 0.0, "zero pattern must return DI=0 (got {di})");
    }

    // ── calculate_beam_pattern ────────────────────────────────────────────────

    /// Beam pattern output is normalised to [0, 1] and has correct angular shape.
    ///
    /// Uses 30° angular resolution to keep the test fast (12×6 direction grid).
    #[test]
    fn calculate_beam_pattern_normalised_and_correct_shape() {
        let grid = Grid::new(4, 4, 4, 1e-3, 1e-3, 1e-3).unwrap();
        let mut field = Array3::<f64>::zeros((4, 4, 4));
        field[[2, 2, 2]] = 1000.0;

        let cfg = BeamPatternConfig {
            frequency: 1e6,
            sound_speed: 1500.0,
            far_field_method: FarFieldMethod::Fraunhofer,
            angular_resolution: PI / 6.0, // 30° → n_theta=12, n_phi=6
        };

        let pattern = calculate_beam_pattern(field.view(), &grid, &cfg).unwrap();

        let n_theta = ((2.0 * PI) / (PI / 6.0)) as usize;
        let n_phi = (PI / (PI / 6.0)) as usize;
        assert_eq!(
            pattern.dim(),
            (n_theta, n_phi),
            "pattern shape must match angular sampling"
        );
        assert!(
            pattern.iter().all(|&v| v >= 0.0 && v <= 1.0 + 1e-10),
            "all pattern values must be in [0, 1]"
        );
        let max_val = pattern.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            (max_val - 1.0).abs() < 1e-10,
            "pattern maximum must be 1.0 (got {max_val:.6})"
        );
    }
}
