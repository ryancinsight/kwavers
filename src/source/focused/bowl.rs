//! Bowl transducer implementation
//!
//! Provides focused bowl transducer geometry and source generation.

use crate::{
    constants::medium_properties::WATER_SOUND_SPEED,
    error::{KwaversError, KwaversResult, ValidationError},
    grid::Grid,
};
use ndarray::Array3;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Configuration for a focused bowl transducer
#[derive(Debug, Clone))]
pub struct BowlConfig {
    /// Radius of curvature (m)
    pub radius_of_curvature: f64,

    /// Diameter of the bowl aperture (m)
    pub diameter: f64,

    /// Center position [x, y, z] (m)
    pub center: [f64; 3],

    /// Focus position [x, y, z] (m)
    pub focus: [f64; 3],

    /// Operating frequency (Hz)
    pub frequency: f64,

    /// Source amplitude (Pa)
    pub amplitude: f64,

    /// Phase delay (radians)
    pub phase: f64,

    /// Element size for discretization (m)
    pub element_size: Option<f64>,

    /// Apply directivity weighting
    pub apply_directivity: bool,
}

impl Default for BowlConfig {
    fn default() -> Self {
        Self {
            radius_of_curvature: 0.064, // 64mm
            diameter: 0.064,            // 64mm
            center: [0.0, 0.0, 0.0],
            focus: [0.0, 0.0, 0.064], // Focus at radius
            frequency: 1e6,           // 1 MHz
            amplitude: 1e6,           // 1 MPa
            phase: 0.0,
            element_size: None,
            apply_directivity: true,
        }
    }
}

/// Focused bowl transducer (makeBowl equivalent)
#[derive(Debug))]
pub struct BowlTransducer {
    pub(crate) config: BowlConfig,
    /// Discretized element positions
    pub(crate) element_positions: Vec<[f64; 3]>,
    /// Element normals for directivity
    pub(crate) element_normals: Vec<[f64; 3]>,
    /// Element areas for weighting
    pub(crate) element_areas: Vec<f64>,
}

impl BowlTransducer {
    /// Create a new bowl transducer
    pub fn new(config: BowlConfig) -> KwaversResult<Self> {
        // Validate configuration
        if config.diameter > 2.0 * config.radius_of_curvature {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "diameter".to_string(),
                value: config.diameter.to_string(),
                constraint: "Diameter cannot exceed 2 * radius_of_curvature".to_string(),
            }));
        }

        // Calculate element size if not provided
        let element_size = config.element_size.unwrap_or_else(|| {
            // Use lambda/4 as default element size
            let speed_of_sound = WATER_SOUND_SPEED;
            let wavelength = speed_of_sound / config.frequency;
            wavelength / 4.0
        });

        // Discretize bowl surface
        let (positions, normals, areas) = Self::discretize_bowl(&config, element_size)?;

        Ok(Self {
            config,
            element_positions: positions,
            element_normals: normals,
            element_areas: areas,
        })
    }

    /// Discretize the bowl surface into elements
    fn discretize_bowl(
        config: &BowlConfig,
        element_size: f64,
    ) -> KwaversResult<(Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<f64>)> {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut areas = Vec::new();

        // Calculate bowl parameters
        let r = config.radius_of_curvature;
        let a = config.diameter / 2.0;
        let h = r - (r * r - a * a).sqrt(); // Height of spherical cap

        // Angular extent of the bowl
        let theta_max = (a / r).asin();

        // Number of angular divisions
        let n_theta = ((2.0 * PI * r * theta_max) / element_size).ceil() as usize;
        let n_phi = ((2.0 * PI * a) / element_size).ceil() as usize;

        // Generate elements using spherical coordinates
        for i in 0..n_theta {
            let theta = (i as f64 + 0.5) * theta_max / n_theta as f64;
            let r_ring = r * theta.sin();
            let z_ring = r * (1.0 - theta.cos());

            // Number of elements in this ring
            let n_ring = ((2.0 * PI * r_ring) / element_size).ceil() as usize;

            for j in 0..n_ring {
                let phi = 2.0 * PI * j as f64 / n_ring as f64;

                // Element position relative to bowl center
                let x = r_ring * phi.cos();
                let y = r_ring * phi.sin();
                let z = z_ring;

                // Transform to global coordinates
                let pos = [
                    config.center[0] + x,
                    config.center[1] + y,
                    config.center[2] + z,
                ];

                // Calculate normal (points inward toward focus)
                let norm_vec = [
                    config.focus[0] - pos[0],
                    config.focus[1] - pos[1],
                    config.focus[2] - pos[2],
                ];
                let norm_mag =
                    (norm_vec[0].powi(2) + norm_vec[1].powi(2) + norm_vec[2].powi(2)).sqrt();
                let normal = [
                    norm_vec[0] / norm_mag,
                    norm_vec[1] / norm_mag,
                    norm_vec[2] / norm_mag,
                ];

                // Calculate element area (approximate)
                let dtheta = theta_max / n_theta as f64;
                let dphi = 2.0 * PI / n_ring as f64;
                let area = r * r * theta.sin() * dtheta * dphi;

                positions.push(pos);
                normals.push(normal);
                areas.push(area);
            }
        }

        Ok((positions, normals, areas))
    }

    /// Generate source distribution on grid
    pub fn generate_source(&self, grid: &Grid, time: f64) -> KwaversResult<Array3<f64>> {
        let mut source = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let omega = 2.0 * PI * self.config.frequency;

        // Calculate phase delays for focusing
        let focus_delays = self.calculate_focus_delays();

        // Use parallel processing for efficiency
        let source_slice = source.as_slice_mut().unwrap();
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let dx = grid.dx;

        source_slice
            .par_chunks_mut(nx * ny)
            .enumerate()
            .for_each(|(iz, chunk)| {
                for iy in 0..ny {
                    for ix in 0..nx {
                        let idx = iy * nx + ix;
                        if idx < chunk.len() {
                            // Grid point position
                            let x = ix as f64 * dx;
                            let y = iy as f64 * dx;
                            let z = iz as f64 * dx;

                            // Accumulate contributions from all elements
                            let mut pressure = 0.0;

                            for (i, &pos) in self.element_positions.iter().enumerate() {
                                // Distance from element to grid point
                                let r = ((x - pos[0]).powi(2)
                                    + (y - pos[1]).powi(2)
                                    + (z - pos[2]).powi(2))
                                .sqrt();

                                if r > 0.0 {
                                    // Apply directivity if enabled
                                    let directivity = if self.config.apply_directivity {
                                        self.calculate_directivity(i, [x, y, z])
                                    } else {
                                        1.0
                                    };

                                    // Phase with focusing delay
                                    let phase =
                                        omega * (time - focus_delays[i]) + self.config.phase;

                                    // Pressure contribution with spherical spreading
                                    let element_pressure = self.config.amplitude
                                        * self.element_areas[i]
                                        * directivity
                                        * phase.sin()
                                        / (4.0 * PI * r);

                                    pressure += element_pressure;
                                }
                            }

                            chunk[idx] = pressure;
                        }
                    }
                }
            });

        Ok(source)
    }

    /// Calculate time delays for focusing
    pub(crate) fn calculate_focus_delays(&self) -> Vec<f64> {
        let speed_of_sound = WATER_SOUND_SPEED;

        self.element_positions
            .iter()
            .map(|&pos| {
                let distance = ((self.config.focus[0] - pos[0]).powi(2)
                    + (self.config.focus[1] - pos[1]).powi(2)
                    + (self.config.focus[2] - pos[2]).powi(2))
                .sqrt();
                distance / speed_of_sound
            })
            .collect()
    }

    /// Calculate directivity for an element
    fn calculate_directivity(&self, element_idx: usize, target: [f64; 3]) -> f64 {
        let pos = self.element_positions[element_idx];
        let normal = self.element_normals[element_idx];

        // Vector from element to target
        let dir = [target[0] - pos[0], target[1] - pos[1], target[2] - pos[2];
        let dir_mag = (dir[0].powi(2) + dir[1].powi(2) + dir[2].powi(2)).sqrt();

        if dir_mag > 0.0 {
            // Cosine of angle between normal and direction
            let cos_theta =
                (normal[0] * dir[0] + normal[1] * dir[1] + normal[2] * dir[2]) / dir_mag;

            // Cosine directivity pattern
            if cos_theta > 0.0 {
                cos_theta
            } else {
                0.0
            }
        } else {
            1.0
        }
    }

    /// Get analytical solution using O'Neil's formula
    ///
    /// O'Neil's solution provides the on-axis pressure field for a spherically
    /// focused transducer. The formula involves complex exponentials representing
    /// the phase accumulation from different parts of the curved surface.
    ///
    /// Reference: O'Neil, H. T. (1949). "Theory of focusing radiators."
    /// The Journal of the Acoustical Society of America, 21(5), 516-526.
    pub fn oneil_solution(&self, z: f64, time: f64) -> f64 {
        // O'Neil's solution for on-axis pressure of a focused bowl transducer
        let r = self.config.radius_of_curvature;
        let a = self.config.diameter / 2.0;
        let k = 2.0 * PI * self.config.frequency / WATER_SOUND_SPEED; // Wave number
        let omega = 2.0 * PI * self.config.frequency; // Angular frequency
        let c = WATER_SOUND_SPEED; // Speed of sound in water

        // Geometric parameters
        // h is the height of the spherical cap
        let h = r - (r * r - a * a).sqrt();

        // Distance from transducer center to field point
        let d1 = z.abs();
        // Distance from edge of spherical cap to field point
        let d2 = ((z - (r - h)).powi(2) + a * a).sqrt();

        // O'Neil's formula for on-axis pressure
        // The pressure is the result of interference between waves from the center
        // and edge of the bowl, accounting for their phase difference
        let phase_diff = k * (d2 - d1);

        // Complex pressure amplitude using proper formulation
        // P(z,t) = P0 * |exp(ikd1) - exp(ikd2)| * exp(i*omega*t) / (2*d1)
        // For magnitude: |exp(ikd1) - exp(ikd2)| = 2*|sin(phase_diff/2)|
        let p_amplitude = if d1 > 0.0 {
            self.config.amplitude * 2.0 * (phase_diff / 2.0).sin().abs() / d1
        } else {
            // At z=0, use limiting value
            self.config.amplitude * k * h
        };

        // Add time-varying component

        p_amplitude * (omega * time - k * d1 + self.config.phase).sin()
    }
}
