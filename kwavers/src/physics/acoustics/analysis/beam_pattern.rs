//! Beam pattern analysis and directivity calculations

use super::validation::{invalid_parameter, validate_pressure_field_domain};
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::core::constants::numerical::MHZ_TO_HZ;
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
            frequency: MHZ_TO_HZ,               // 1 MHz
            sound_speed: SOUND_SPEED_WATER_SIM, // Default sound speed
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
#[must_use]
pub fn calculate_beam_pattern(
    pressure_field: ArrayView3<f64>,
    grid: &Grid,
    config: &BeamPatternConfig,
) -> KwaversResult<Array2<f64>> {
    validate_beam_pattern_config(config)?;
    validate_pressure_field_domain(pressure_field, grid)?;

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

/// Calculate directivity index.
///
/// For a pressure-amplitude beam pattern `B`, acoustic intensity is
/// proportional to `|B|²`, so `DI = 10 log10(max |B|² / mean |B|²)`.
#[must_use]
pub fn calculate_directivity(beam_pattern: &Array2<f64>) -> f64 {
    let mut max_power = 0.0f64;
    let mut power_sum = 0.0f64;
    let mut sample_count = 0usize;

    for &sample in beam_pattern {
        if !sample.is_finite() {
            return 0.0;
        }

        let magnitude = sample.abs();
        let power = magnitude * magnitude;
        max_power = max_power.max(power);
        power_sum += power;
        sample_count += 1;
    }

    if sample_count == 0 {
        return 0.0;
    }

    let mean_power = power_sum / sample_count as f64;

    if max_power > 0.0 && mean_power > 0.0 {
        10.0 * (max_power / mean_power).log10()
    } else {
        0.0
    }
}

fn validate_beam_pattern_config(config: &BeamPatternConfig) -> KwaversResult<()> {
    validate_positive_finite("frequency", config.frequency)?;
    validate_positive_finite("sound_speed", config.sound_speed)?;
    validate_positive_finite("angular_resolution", config.angular_resolution)?;

    if config.angular_resolution > PI {
        return Err(invalid_parameter(
            "angular_resolution",
            config.angular_resolution,
            "must not exceed pi radians because at least one polar sample is required",
        ));
    }

    Ok(())
}

fn validate_positive_finite(parameter: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid_parameter(
            parameter,
            value,
            "must be positive and finite",
        ))
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
        assert!((cfg.frequency - MHZ_TO_HZ).abs() < 1.0, "frequency must be 1 MHz");
        assert!(
            (cfg.sound_speed - SOUND_SPEED_WATER_SIM).abs() < 1e-10,
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

    /// Directivity uses intensity, proportional to squared pressure magnitude,
    /// before averaging. Signed amplitudes must not cancel the denominator.
    #[test]
    fn calculate_directivity_uses_intensity_average() {
        let pattern = Array2::from_shape_vec((1, 2), vec![1.0, -0.5]).unwrap();
        let di = calculate_directivity(&pattern);
        let expected = 10.0_f64 * (1.0_f64 / 0.625_f64).log10();

        assert!(
            (di - expected).abs() < 1e-12,
            "intensity-averaged DI expected {expected}, got {di}"
        );
    }

    /// Nonfinite samples have no finite directivity interpretation in this
    /// infallible scalar API.
    #[test]
    fn calculate_directivity_rejects_nonfinite_samples() {
        let pattern = Array2::from_shape_vec((1, 2), vec![1.0, f64::NAN]).unwrap();
        let di = calculate_directivity(&pattern);

        assert_eq!(di, 0.0, "nonfinite pattern must return zero DI");
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
            frequency: MHZ_TO_HZ,
            sound_speed: SOUND_SPEED_WATER_SIM,
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

    #[test]
    fn calculate_beam_pattern_rejects_invalid_config_domain() {
        let grid = Grid::new(4, 4, 4, 1e-3, 1e-3, 1e-3).unwrap();
        let field = Array3::<f64>::zeros((4, 4, 4));
        let cfg = BeamPatternConfig {
            frequency: 0.0,
            sound_speed: SOUND_SPEED_WATER_SIM,
            far_field_method: FarFieldMethod::Fraunhofer,
            angular_resolution: PI / 6.0,
        };

        let err = calculate_beam_pattern(field.view(), &grid, &cfg).unwrap_err();
        let message = err.to_string();

        assert!(message.contains("frequency"), "unexpected error: {message}");
        assert!(
            message.contains("positive and finite"),
            "domain reason must be reported: {message}"
        );
    }

    #[test]
    fn calculate_beam_pattern_rejects_shape_mismatch() {
        let grid = Grid::new(4, 4, 4, 1e-3, 1e-3, 1e-3).unwrap();
        let field = Array3::<f64>::zeros((3, 4, 4));
        let cfg = BeamPatternConfig {
            angular_resolution: PI / 6.0,
            ..BeamPatternConfig::default()
        };

        let err = calculate_beam_pattern(field.view(), &grid, &cfg).unwrap_err();
        let message = err.to_string();

        assert!(
            message.contains("Dimension mismatch"),
            "unexpected error: {message}"
        );
        assert!(message.contains("(4, 4, 4)"), "expected shape: {message}");
        assert!(message.contains("(3, 4, 4)"), "actual shape: {message}");
    }

    #[test]
    fn calculate_beam_pattern_rejects_nonfinite_pressure() {
        let grid = Grid::new(4, 4, 4, 1e-3, 1e-3, 1e-3).unwrap();
        let mut field = Array3::<f64>::zeros((4, 4, 4));
        field[[1, 2, 3]] = f64::INFINITY;
        let cfg = BeamPatternConfig {
            angular_resolution: PI / 6.0,
            ..BeamPatternConfig::default()
        };

        let err = calculate_beam_pattern(field.view(), &grid, &cfg).unwrap_err();
        let message = err.to_string();

        assert!(message.contains("nonfinite"), "unexpected error: {message}");
        assert!(
            message.contains("[1, 2, 3]"),
            "sample index must be reported: {message}"
        );
    }
}
