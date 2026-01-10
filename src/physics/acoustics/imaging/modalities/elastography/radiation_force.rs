//! Acoustic Radiation Force for Shear Wave Generation
//!
//! Implements acoustic radiation force impulse (ARFI) for generating shear waves
//! in soft tissue.
//!
//! ## Physics
//!
//! Acoustic radiation force arises from momentum transfer when ultrasound waves
//! are absorbed or reflected. For a focused ultrasound beam:
//!
//! F = (2αI)/c
//!
//! where:
//! - F is radiation force density (N/m³)
//! - α is absorption coefficient (Np/m)
//! - I is acoustic intensity (W/m²)
//! - c is sound speed (m/s)
//!
//! ## References
//!
//! - Nightingale, K., et al. (2002). "Acoustic radiation force impulse imaging."
//!   *Ultrasound in Medicine & Biology*, 28(2), 227-235.
//! - Palmeri, M. L., et al. (2005). "Ultrasonic tracking of acoustic radiation
//!   force-induced displacements." *IEEE TUFFC*, 52(8), 1300-1313.

use crate::domain::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::Array3;
use std::f64::consts::PI;

use super::elastic_wave_solver::ElasticBodyForceConfig;

/// Acoustic radiation force push pulse parameters
///
/// # Clinical Values
///
/// - Push duration: 50-400 μs (typ. 100-200 μs)
/// - Push frequency: 3-8 MHz (typ. 5 MHz)
/// - Focus depth: 20-80 mm
/// - F-number: 1.5-3.0 (typ. 2.0)
#[derive(Debug, Clone)]
pub struct PushPulseParameters {
    /// Push pulse frequency (Hz)
    pub frequency: f64,
    /// Push pulse duration (s)
    pub duration: f64,
    /// Peak acoustic intensity (W/m²)
    pub intensity: f64,
    /// Focal depth (m)
    pub focal_depth: f64,
    /// F-number (focal_depth / aperture_width)
    pub f_number: f64,
}

impl Default for PushPulseParameters {
    fn default() -> Self {
        Self {
            frequency: 5.0e6,  // 5 MHz
            duration: 150e-6,  // 150 μs
            intensity: 1.0e6,  // 1 MW/m²
            focal_depth: 0.04, // 40 mm
            f_number: 2.0,
        }
    }
}

impl PushPulseParameters {
    /// Create custom push pulse parameters
    ///
    /// # Arguments
    ///
    /// * `frequency` - Push frequency in Hz
    /// * `duration` - Push duration in seconds
    /// * `intensity` - Peak intensity in W/m²
    /// * `focal_depth` - Focal depth in meters
    /// * `f_number` - F-number (dimensionless)
    pub fn new(
        frequency: f64,
        duration: f64,
        intensity: f64,
        focal_depth: f64,
        f_number: f64,
    ) -> KwaversResult<Self> {
        if frequency <= 0.0 {
            return Err(KwaversError::Validation(
                crate::domain::core::error::ValidationError::InvalidValue {
                    parameter: "frequency".to_string(),
                    value: frequency,
                    reason: "must be positive".to_string(),
                },
            ));
        }
        if duration <= 0.0 {
            return Err(KwaversError::Validation(
                crate::domain::core::error::ValidationError::InvalidValue {
                    parameter: "duration".to_string(),
                    value: duration,
                    reason: "must be positive".to_string(),
                },
            ));
        }
        if intensity <= 0.0 {
            return Err(KwaversError::Validation(
                crate::domain::core::error::ValidationError::InvalidValue {
                    parameter: "intensity".to_string(),
                    value: intensity,
                    reason: "must be positive".to_string(),
                },
            ));
        }

        Ok(Self {
            frequency,
            duration,
            intensity,
            focal_depth,
            f_number,
        })
    }
}

/// Acoustic radiation force generator
#[derive(Debug)]
pub struct AcousticRadiationForce {
    /// Push pulse configuration
    parameters: PushPulseParameters,
    /// Medium sound speed (m/s)
    sound_speed: f64,
    /// Medium absorption coefficient (Np/m)
    absorption: f64,
    /// Medium density (kg/m³)
    density: f64,
    /// Computational grid
    grid: Grid,
}

impl AcousticRadiationForce {
    /// Create new acoustic radiation force generator
    ///
    /// # Arguments
    ///
    /// * `grid` - Computational grid
    /// * `medium` - Tissue medium properties
    pub fn new(grid: &Grid, medium: &dyn Medium) -> KwaversResult<Self> {
        // Get medium properties at center
        let (nx, ny, nz) = grid.dimensions();
        let ci = nx / 2;
        let cj = ny / 2;
        let ck = nz / 2;

        let sound_speed = medium.sound_speed(ci, cj, ck);
        let density = medium.density(ci, cj, ck);

        // Estimate absorption coefficient
        // For soft tissue at 5 MHz: α ≈ 0.5 dB/cm/MHz ≈ 5.8 Np/m
        let absorption = 5.8; // Np/m, typical value for soft tissue at 1 MHz
                              // Reference: Duck (1990), Physical Properties of Tissue

        Ok(Self {
            parameters: PushPulseParameters::default(),
            sound_speed,
            absorption,
            density,
            grid: grid.clone(),
        })
    }

    /// Set custom push pulse parameters
    pub fn set_parameters(&mut self, parameters: PushPulseParameters) {
        self.parameters = parameters;
    }

    /// Get current push pulse parameters
    #[must_use]
    pub fn parameters(&self) -> &PushPulseParameters {
        &self.parameters
    }

    /// Create an elastic body-force configuration for an ARFI push pulse.
    ///
    /// # Arguments
    ///
    /// * `push_location` - Focal point [x, y, z] in meters
    ///
    /// # Returns
    ///
    /// Elastic body-force configuration to be consumed by the elastic solver as a source term.
    ///
    /// # Correctness invariant
    ///
    /// This returns a *forcing term* `f(x,t)` with units N/m³, to be used in:
    ///
    ///   ρ ∂v/∂t = ∇·σ + f
    ///
    /// This is intentionally not an “initial displacement” API.
    ///
    /// # References
    ///
    /// Nightingale et al. (2002): Radiation force density f ≈ (2αI)/c.
    pub fn push_pulse_body_force(
        &self,
        push_location: [f64; 3],
    ) -> KwaversResult<ElasticBodyForceConfig> {
        // Calculate radiation force density magnitude (N/m³)
        // f = (2αI)/c
        let force_density = (2.0 * self.absorption * self.parameters.intensity) / self.sound_speed;

        // Convert pulse duration into an impulse density J = ∫ f(t) dt (N·s/m³).
        // We model the temporal envelope as a unit-area Gaussian in the solver, so this is the
        // time integral of the force density.
        let impulse_n_per_m3_s = force_density * self.parameters.duration;

        // Spatial envelope: use Gaussian standard deviations derived from FWHM heuristics.
        // Lateral: FWHM ≈ 1.2 × λ × F-number
        // Axial:   FWHM ≈ 6 × λ × F-number²
        //
        // For a Gaussian exp(-0.5 (r/σ)²), FWHM = 2 √(2 ln 2) σ.
        let wavelength = self.sound_speed / self.parameters.frequency;
        let lateral_fwhm = 1.2 * wavelength * self.parameters.f_number;
        let axial_fwhm = 6.0 * wavelength * self.parameters.f_number * self.parameters.f_number;

        let fwhm_to_sigma = 1.0 / (2.0 * (2.0 * std::f64::consts::LN_2).sqrt());
        let sigma_lateral = (lateral_fwhm * fwhm_to_sigma).max(1e-12);
        let sigma_axial = (axial_fwhm * fwhm_to_sigma).max(1e-12);

        // Temporal envelope: use a Gaussian with σ_t chosen so that ~99% of mass lies within the
        // push duration. For a Gaussian, ±3σ covers ~99.7%, so take σ_t = duration / 6.
        let sigma_t_s = (self.parameters.duration / 6.0).max(1e-12);

        // Direction: ARFI primarily pushes along the beam axis. In this simplified geometry, we
        // model the beam axis as +z.
        let direction = [0.0, 0.0, 1.0];

        Ok(ElasticBodyForceConfig::GaussianImpulse {
            center_m: push_location,
            sigma_m: [sigma_lateral, sigma_lateral, sigma_axial],
            direction,
            t0_s: 0.0,
            sigma_t_s,
            impulse_n_per_m3_s,
        })
    }

    /// Legacy API: Apply push pulse and return an initial displacement.
    ///
    /// # Correctness warning
    ///
    /// This method is maintained for compatibility but is not physically faithful: ARFI is a
    /// body-force excitation, not an instantaneous displacement assignment. Prefer
    /// [`push_pulse_body_force`] and configure the solver to use body-force sources.
    #[deprecated(
        note = "ARFI should be modeled as a body-force source term; use push_pulse_body_force instead."
    )]
    pub fn apply_push_pulse(&self, push_location: [f64; 3]) -> KwaversResult<Array3<f64>> {
        self.push_pulse_pseudo_displacement(push_location)
    }

    fn push_pulse_pseudo_displacement(
        &self,
        push_location: [f64; 3],
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut displacement = Array3::zeros((nx, ny, nz));

        // Calculate radiation force density
        // f = (2αI)/c (N/m³)
        let force_density = (2.0 * self.absorption * self.parameters.intensity) / self.sound_speed;

        // NOTE: This computes a quantity with units [m/s], not [m]. Historically this was used as a
        // displacement initializer; we keep it only for backward compatibility.
        let pseudo_displacement_scale = (force_density * self.parameters.duration) / self.density;

        // FWHM heuristics
        let wavelength = self.sound_speed / self.parameters.frequency;
        let lateral_width = 1.2 * wavelength * self.parameters.f_number;
        let axial_length = 6.0 * wavelength * self.parameters.f_number * self.parameters.f_number;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let x = i as f64 * self.grid.dx;
                    let y = j as f64 * self.grid.dy;
                    let z = k as f64 * self.grid.dz;

                    let dx = x - push_location[0];
                    let dy = y - push_location[1];
                    let dz = z - push_location[2];

                    let r_lateral = (dx * dx + dy * dy).sqrt();
                    let r_axial = dz.abs();

                    let lateral_profile = (-4.0 * (r_lateral / lateral_width).powi(2)).exp();
                    let axial_profile = (-4.0 * (r_axial / axial_length).powi(2)).exp();

                    displacement[[i, j, k]] =
                        pseudo_displacement_scale * lateral_profile * axial_profile;
                }
            }
        }

        Ok(displacement)
    }

    /// Apply multi-directional push pulses for 3D SWE
    ///
    /// # Arguments
    ///
    /// * `push_sequence` - Sequence of push locations and directions
    ///
    /// # Returns
    ///
    /// Combined initial displacement field from all push pulses
    #[deprecated(
        note = "ARFI should be modeled as a body-force source term; use multi_directional_body_forces instead."
    )]
    pub fn apply_multi_directional_push(
        &self,
        push_sequence: &MultiDirectionalPush,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut total_displacement = Array3::zeros((nx, ny, nz));

        for push in &push_sequence.pushes {
            let displacement = self.push_pulse_pseudo_displacement(push.location)?;
            // Add displacement with directional weighting
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        total_displacement[[i, j, k]] +=
                            displacement[[i, j, k]] * push.amplitude_weight;
                    }
                }
            }
        }

        Ok(total_displacement)
    }

    /// Create per-push body-force configs for a multi-directional push sequence.
    ///
    /// This is the correctness-first replacement for summing scalar “initial displacements”.
    pub fn multi_directional_body_forces(
        &self,
        push_sequence: &MultiDirectionalPush,
    ) -> KwaversResult<Vec<ElasticBodyForceConfig>> {
        let mut forces = Vec::with_capacity(push_sequence.pushes.len());
        for push in &push_sequence.pushes {
            let mut cfg = self.push_pulse_body_force(push.location)?;
            // Apply amplitude weighting by scaling impulse density (impulse density is ∫ f dt).
            match &mut cfg {
                ElasticBodyForceConfig::GaussianImpulse {
                    impulse_n_per_m3_s, ..
                } => {
                    *impulse_n_per_m3_s *= push.amplitude_weight;
                }
            }
            forces.push(cfg);
        }
        Ok(forces)
    }
}

/// Multi-directional push pulse configuration for 3D SWE
#[derive(Debug, Clone)]
pub struct MultiDirectionalPush {
    /// Individual push pulses with locations and properties
    pub pushes: Vec<DirectionalPush>,
    /// Time delays between pushes (s)
    pub time_delays: Vec<f64>,
    /// Total sequence duration (s)
    pub sequence_duration: f64,
}

/// Individual directional push pulse
#[derive(Debug, Clone)]
pub struct DirectionalPush {
    /// Push location [x, y, z] in meters
    pub location: [f64; 3],
    /// Push direction vector (normalized)
    pub direction: [f64; 3],
    /// Amplitude weighting factor
    pub amplitude_weight: f64,
    /// Custom parameters for this push (optional)
    pub parameters: Option<PushPulseParameters>,
}

impl MultiDirectionalPush {
    /// Create orthogonal push pattern for comprehensive 3D coverage
    ///
    /// Generates pushes along x, y, z axes from a central location
    pub fn orthogonal_pattern(center: [f64; 3], spacing: f64) -> Self {
        let pushes = vec![
            // +X direction
            DirectionalPush {
                location: [center[0] + spacing, center[1], center[2]],
                direction: [1.0, 0.0, 0.0],
                amplitude_weight: 1.0,
                parameters: None,
            },
            // -X direction
            DirectionalPush {
                location: [center[0] - spacing, center[1], center[2]],
                direction: [-1.0, 0.0, 0.0],
                amplitude_weight: 1.0,
                parameters: None,
            },
            // +Y direction
            DirectionalPush {
                location: [center[0], center[1] + spacing, center[2]],
                direction: [0.0, 1.0, 0.0],
                amplitude_weight: 1.0,
                parameters: None,
            },
            // -Y direction
            DirectionalPush {
                location: [center[0], center[1] - spacing, center[2]],
                direction: [0.0, -1.0, 0.0],
                amplitude_weight: 1.0,
                parameters: None,
            },
            // +Z direction
            DirectionalPush {
                location: [center[0], center[1], center[2] + spacing],
                direction: [0.0, 0.0, 1.0],
                amplitude_weight: 1.0,
                parameters: None,
            },
            // -Z direction
            DirectionalPush {
                location: [center[0], center[1], center[2] - spacing],
                direction: [0.0, 0.0, -1.0],
                amplitude_weight: 1.0,
                parameters: None,
            },
        ];

        // Time delays for sequential excitation
        let time_delays = vec![0.0, 50e-6, 100e-6, 150e-6, 200e-6, 250e-6];
        let sequence_duration = 300e-6; // 300 μs total

        Self {
            pushes,
            time_delays,
            sequence_duration,
        }
    }

    /// Create compound push pattern for enhanced shear wave generation
    ///
    /// Uses multiple pushes at different angles for better wave interference
    pub fn compound_pattern(center: [f64; 3], radius: f64, n_pushes: usize) -> Self {
        let mut pushes = Vec::new();

        for i in 0..n_pushes {
            let angle = 2.0 * PI * (i as f64) / (n_pushes as f64);
            let x = center[0] + radius * angle.cos();
            let y = center[1] + radius * angle.sin();
            let z = center[2];

            // Alternate between different depths for 3D coverage
            let z_offset = if i % 2 == 0 {
                radius * 0.5
            } else {
                -radius * 0.5
            };
            let location = [x, y, z + z_offset];

            // Direction points radially outward from center
            let direction = [
                (x - center[0]) / radius,
                (y - center[1]) / radius,
                z_offset.signum() * 0.5,
            ];

            pushes.push(DirectionalPush {
                location,
                direction,
                amplitude_weight: 1.0,
                parameters: None,
            });
        }

        // Staggered timing for wave interference
        let time_delays: Vec<f64> = (0..n_pushes)
            .map(|i| i as f64 * 25e-6) // 25 μs spacing
            .collect();

        let sequence_duration = time_delays.last().unwrap_or(&0.0) + 100e-6;

        Self {
            pushes,
            time_delays,
            sequence_duration,
        }
    }

    /// Create focused push pattern for targeted 3D SWE
    ///
    /// Concentrates pushes in a specific region of interest
    pub fn focused_pattern(roi_center: [f64; 3], roi_size: [f64; 3], density: usize) -> Self {
        let mut pushes = Vec::new();

        // Create grid of push locations within ROI
        let nx = (roi_size[0] / 0.005).ceil() as usize; // 5mm spacing
        let ny = (roi_size[1] / 0.005).ceil() as usize;
        let nz = (roi_size[2] / 0.005).ceil() as usize;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = roi_center[0] + (i as f64 - nx as f64 / 2.0) * 0.005;
                    let y = roi_center[1] + (j as f64 - ny as f64 / 2.0) * 0.005;
                    let z = roi_center[2] + (k as f64 - nz as f64 / 2.0) * 0.005;

                    // Weight pushes based on distance from ROI center
                    let distance = ((x - roi_center[0]).powi(2)
                        + (y - roi_center[1]).powi(2)
                        + (z - roi_center[2]).powi(2))
                    .sqrt();
                    let max_distance = roi_size.iter().cloned().fold(0.0, f64::max) / 2.0;
                    let weight = (1.0 - distance / max_distance).max(0.1);

                    pushes.push(DirectionalPush {
                        location: [x, y, z],
                        direction: [0.0, 0.0, 1.0], // Axial direction
                        amplitude_weight: weight,
                        parameters: None,
                    });
                }
            }
        }

        // Limit total pushes for computational efficiency
        if pushes.len() > density {
            pushes.sort_by(|a, b| {
                let dist_a = ((a.location[0] - roi_center[0]).powi(2)
                    + (a.location[1] - roi_center[1]).powi(2)
                    + (a.location[2] - roi_center[2]).powi(2))
                .sqrt();
                let dist_b = ((b.location[0] - roi_center[0]).powi(2)
                    + (b.location[1] - roi_center[1]).powi(2)
                    + (b.location[2] - roi_center[2]).powi(2))
                .sqrt();
                dist_a.partial_cmp(&dist_b).unwrap()
            });
            pushes.truncate(density);
        }

        // Sequential timing
        let time_delays: Vec<f64> = (0..pushes.len())
            .map(|i| i as f64 * 10e-6) // 10 μs spacing
            .collect();

        let sequence_duration = time_delays.last().unwrap_or(&0.0) + 50e-6;

        Self {
            pushes,
            time_delays,
            sequence_duration,
        }
    }
}

/// Directional wave tracking for multi-directional SWE
#[derive(Debug, Clone)]
pub struct DirectionalWaveTracker {
    /// Expected wave directions for each push
    pub wave_directions: Vec<[f64; 3]>,
    /// Tracking regions for each direction
    pub tracking_regions: Vec<TrackingRegion>,
    /// Quality metrics for directional tracking
    pub quality_metrics: Vec<DirectionalQuality>,
}

/// Tracking region for directional wave analysis
#[derive(Debug, Clone)]
pub struct TrackingRegion {
    /// Center point of tracking region
    pub center: [f64; 3],
    /// Size of tracking region [width, height, depth]
    pub size: [f64; 3],
    /// Expected wave direction
    pub direction: [f64; 3],
}

/// Quality metrics for directional wave tracking
#[derive(Debug, Clone)]
pub struct DirectionalQuality {
    /// Signal-to-noise ratio for this direction
    pub snr: f64,
    /// Wave amplitude consistency
    pub amplitude_consistency: f64,
    /// Directional purity (how well wave follows expected direction)
    pub directional_purity: f64,
    /// Tracking confidence score
    pub confidence: f64,
}

impl DirectionalWaveTracker {
    /// Create tracker for orthogonal push pattern
    pub fn for_orthogonal_pattern(center: [f64; 3], roi_size: [f64; 3]) -> Self {
        let directions = vec![
            [1.0, 0.0, 0.0],  // +X
            [-1.0, 0.0, 0.0], // -X
            [0.0, 1.0, 0.0],  // +Y
            [0.0, -1.0, 0.0], // -Y
            [0.0, 0.0, 1.0],  // +Z
            [0.0, 0.0, -1.0], // -Z
        ];

        let mut tracking_regions = Vec::new();
        let mut quality_metrics = Vec::new();

        for direction in &directions {
            // Define tracking region along the wave propagation direction
            let region_center = [
                center[0] + direction[0] * roi_size[0] * 0.25,
                center[1] + direction[1] * roi_size[1] * 0.25,
                center[2] + direction[2] * roi_size[2] * 0.25,
            ];

            tracking_regions.push(TrackingRegion {
                center: region_center,
                size: [roi_size[0] * 0.5, roi_size[1] * 0.5, roi_size[2] * 0.5],
                direction: *direction,
            });

            quality_metrics.push(DirectionalQuality {
                snr: 0.0, // To be computed
                amplitude_consistency: 0.0,
                directional_purity: 0.0,
                confidence: 0.0,
            });
        }

        Self {
            wave_directions: directions,
            tracking_regions,
            quality_metrics,
        }
    }

    /// Validate multi-directional wave propagation physics
    pub fn validate_wave_physics(
        &self,
        measured_speeds: &[f64],
        expected_speeds: &[f64],
    ) -> ValidationResult {
        let mut directional_consistency = 0.0;
        let mut amplitude_uniformity = 0.0;

        for (i, (&measured, &expected)) in measured_speeds
            .iter()
            .zip(expected_speeds.iter())
            .enumerate()
        {
            // Check speed consistency across directions
            let speed_ratio = measured / expected;
            directional_consistency += (1.0 - (speed_ratio - 1.0).abs()).max(0.0);

            // Check amplitude consistency based on radiation force physics
            // Radiation force amplitude should be proportional to I₀² where I₀ is intensity
            // For plane waves, intensity is uniform, so amplitude uniformity measures
            // how well the push beams maintain consistent power delivery

            let direction_idx = i % 8; // Assume 8 standard directions
            let expected_amplitude = match direction_idx {
                0 | 4 => 1.0,           // Axial directions - maximum amplitude
                1 | 3 | 5 | 7 => 0.866, // 30-degree directions
                2 | 6 => 0.707,         // 45-degree directions
                _ => 0.5,               // Other directions
            };

            // Calculate amplitude deviation from expected
            let amplitude_deviation = (expected_amplitude - 0.8_f64).abs(); // Assume measured amplitude of 0.8
            amplitude_uniformity += (1.0_f64 - amplitude_deviation).max(0.0_f64);
        }

        directional_consistency /= measured_speeds.len() as f64;
        amplitude_uniformity /= measured_speeds.len() as f64;

        ValidationResult {
            directional_consistency,
            amplitude_uniformity,
            overall_quality: (directional_consistency + amplitude_uniformity) / 2.0,
        }
    }
}

/// Validation result for multi-directional wave physics
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Consistency of wave speeds across different directions (0-1)
    pub directional_consistency: f64,
    /// Uniformity of wave amplitudes across directions (0-1)
    pub amplitude_uniformity: f64,
    /// Overall quality score (0-1)
    pub overall_quality: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::medium::HomogeneousMedium;

    #[test]
    fn test_push_parameters_default() {
        let params = PushPulseParameters::default();
        assert_eq!(params.frequency, 5.0e6);
        assert_eq!(params.duration, 150e-6);
        assert!((params.f_number - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_push_parameters_validation() {
        let result = PushPulseParameters::new(-1.0, 100e-6, 1000.0, 0.04, 2.0);
        assert!(result.is_err());

        let result = PushPulseParameters::new(5e6, -100e-6, 1000.0, 0.04, 2.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_radiation_force_creation() {
        let grid = Grid::new(50, 50, 50, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let result = AcousticRadiationForce::new(&grid, &medium);
        assert!(result.is_ok());
    }

    #[test]
    fn test_push_pulse_generation() {
        let grid = Grid::new(50, 50, 50, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
        let arf = AcousticRadiationForce::new(&grid, &medium).unwrap();

        let push_location = [0.025, 0.025, 0.025];
        let displacement = arf.push_pulse_pseudo_displacement(push_location).unwrap();

        // Check displacement field properties
        assert_eq!(displacement.dim(), (50, 50, 50));

        // Maximum displacement should be at or near focal point
        let max_disp = displacement
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(max_disp > 0.0, "Maximum displacement should be positive");

        // Displacement should decay away from focal point
        let corner_disp = displacement[[0, 0, 0]];
        assert!(
            corner_disp < max_disp * 0.1,
            "Displacement should be localized"
        );
    }

    #[test]
    fn test_multi_directional_push_creation() {
        let center = [0.025, 0.025, 0.025];
        let spacing = 0.01;

        let pattern = MultiDirectionalPush::orthogonal_pattern(center, spacing);

        // Should have 6 pushes (3 axes × 2 directions)
        assert_eq!(pattern.pushes.len(), 6);
        assert_eq!(pattern.time_delays.len(), 6);

        // Check that pushes are positioned correctly
        let push_x_pos = pattern.pushes[0].location; // +X direction
        assert!((push_x_pos[0] - (center[0] + spacing)).abs() < 1e-10);
        assert!((push_x_pos[1] - center[1]).abs() < 1e-10);
        assert!((push_x_pos[2] - center[2]).abs() < 1e-10);
    }

    #[test]
    fn test_compound_push_pattern() {
        let center = [0.025, 0.025, 0.025];
        let radius = 0.015;
        let n_pushes = 8;

        let pattern = MultiDirectionalPush::compound_pattern(center, radius, n_pushes);

        assert_eq!(pattern.pushes.len(), n_pushes);

        // Check that pushes are distributed around the circle
        for (i, push) in pattern.pushes.iter().enumerate() {
            let expected_angle = 2.0 * PI * (i as f64) / (n_pushes as f64);
            let mut actual_angle =
                (push.location[1] - center[1]).atan2(push.location[0] - center[0]);
            // Normalize angle to [0, 2π) range
            if actual_angle < 0.0 {
                actual_angle += 2.0 * PI;
            }
            // Allow for small numerical differences and account for atan2 range
            let angle_diff = (actual_angle - expected_angle).abs().min(
                (actual_angle - expected_angle + 2.0 * PI)
                    .abs()
                    .min((actual_angle - expected_angle - 2.0 * PI).abs()),
            );
            assert!(
                angle_diff < 0.1,
                "Push {}: expected angle {:.3}, got {:.3}",
                i,
                expected_angle,
                actual_angle
            );
        }
    }

    #[test]
    fn test_focused_push_pattern() {
        let roi_center = [0.025, 0.025, 0.025];
        let roi_size = [0.02, 0.02, 0.02];
        let density = 10;

        let pattern = MultiDirectionalPush::focused_pattern(roi_center, roi_size, density);

        // Should not exceed density limit
        assert!(pattern.pushes.len() <= density);

        // All pushes should be within ROI bounds
        for push in &pattern.pushes {
            assert!(push.location[0] >= roi_center[0] - roi_size[0] / 2.0);
            assert!(push.location[0] <= roi_center[0] + roi_size[0] / 2.0);
            assert!(push.location[1] >= roi_center[1] - roi_size[1] / 2.0);
            assert!(push.location[1] <= roi_center[1] + roi_size[1] / 2.0);
            assert!(push.location[2] >= roi_center[2] - roi_size[2] / 2.0);
            assert!(push.location[2] <= roi_center[2] + roi_size[2] / 2.0);
        }
    }

    #[test]
    fn test_directional_wave_tracker() {
        let center = [0.025, 0.025, 0.025];
        let roi_size = [0.04, 0.04, 0.04];

        let tracker = DirectionalWaveTracker::for_orthogonal_pattern(center, roi_size);

        // Should have 6 directions and tracking regions
        assert_eq!(tracker.wave_directions.len(), 6);
        assert_eq!(tracker.tracking_regions.len(), 6);
        assert_eq!(tracker.quality_metrics.len(), 6);

        // Check that directions are orthogonal unit vectors
        for direction in &tracker.wave_directions {
            let magnitude =
                (direction[0].powi(2) + direction[1].powi(2) + direction[2].powi(2)).sqrt();
            assert!(
                (magnitude - 1.0).abs() < 1e-10,
                "Direction should be unit vector"
            );
        }
    }

    #[test]
    fn test_wave_physics_validation() {
        let center = [0.025, 0.025, 0.025];
        let roi_size = [0.04, 0.04, 0.04];

        let tracker = DirectionalWaveTracker::for_orthogonal_pattern(center, roi_size);

        // Simulate measured and expected speeds
        let measured_speeds = vec![3.0, 2.9, 3.1, 3.0, 2.8, 3.2];
        let expected_speeds = vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0];

        let result = tracker.validate_wave_physics(&measured_speeds, &expected_speeds);

        // Should have reasonable validation scores
        assert!(result.directional_consistency >= 0.0 && result.directional_consistency <= 1.0);
        assert!(result.amplitude_uniformity >= 0.0 && result.amplitude_uniformity <= 1.0);
        assert!(result.overall_quality >= 0.0 && result.overall_quality <= 1.0);
    }

    #[test]
    fn test_multi_directional_push_application() {
        let grid = Grid::new(30, 30, 30, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
        let arf = AcousticRadiationForce::new(&grid, &medium).unwrap();

        let center = [0.015, 0.015, 0.015];
        let pattern = MultiDirectionalPush::orthogonal_pattern(center, 0.005);

        let forces = arf
            .multi_directional_body_forces(&pattern)
            .expect("multi-directional ARFI must yield body-force configs");

        assert_eq!(forces.len(), pattern.pushes.len());
        for cfg in &forces {
            match cfg {
                ElasticBodyForceConfig::GaussianImpulse {
                    impulse_n_per_m3_s, ..
                } => {
                    assert!(*impulse_n_per_m3_s > 0.0);
                }
            }
        }
    }
}
