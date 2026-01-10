//! Arc source implementation for 2D focused transducers
//!
//! Provides arc-shaped transducer geometry for 2D simulations.

use crate::{
    domain::core::{constants::SOUND_SPEED_WATER, error::KwaversResult},
    domain::grid::Grid,
};
use ndarray::{s, Array2, Array3, Zip};
use std::f64::consts::PI;

/// Configuration for an arc source (2D focused transducer)
#[derive(Debug, Clone)]
pub struct ArcConfig {
    /// Radius of curvature (m)
    pub radius: f64,

    /// Arc angle (radians)
    pub arc_angle: f64,

    /// Center position [x, y] (m)
    pub center: [f64; 2],

    /// Orientation angle (radians)
    pub orientation: f64,

    /// Operating frequency (Hz)
    pub frequency: f64,

    /// Source amplitude (Pa)
    pub amplitude: f64,

    /// Element spacing (m)
    pub element_spacing: Option<f64>,
}

impl Default for ArcConfig {
    fn default() -> Self {
        Self {
            radius: 0.05,        // 50mm
            arc_angle: PI / 3.0, // 60 degrees
            center: [0.0, 0.0],
            orientation: 0.0,
            frequency: 1e6, // 1 MHz
            amplitude: 1e6, // 1 MPa
            element_spacing: None,
        }
    }
}

/// Arc source for 2D simulations (makeArc equivalent)
#[derive(Debug)]
pub struct ArcSource {
    config: ArcConfig,
    /// Discretized element positions
    element_positions: Vec<[f64; 2]>,
    /// Element weights
    element_weights: Vec<f64>,
}

impl ArcSource {
    /// Create a new arc source
    pub fn new(config: ArcConfig) -> KwaversResult<Self> {
        // Calculate element spacing if not provided
        let element_spacing = config.element_spacing.unwrap_or_else(|| {
            let speed_of_sound = SOUND_SPEED_WATER;
            let wavelength = speed_of_sound / config.frequency;
            wavelength / 4.0
        });

        // Number of elements
        let arc_length = config.radius * config.arc_angle;
        let n_elements = (arc_length / element_spacing).ceil() as usize;

        // Generate element positions
        let mut positions = Vec::with_capacity(n_elements);
        let mut weights = Vec::with_capacity(n_elements);

        for i in 0..n_elements {
            // Angle for this element
            let theta =
                -config.arc_angle / 2.0 + (i as f64 + 0.5) * config.arc_angle / n_elements as f64;
            let rotated_theta = theta + config.orientation;

            // Position
            let x = config.center[0] + config.radius * rotated_theta.cos();
            let y = config.center[1] + config.radius * rotated_theta.sin();

            positions.push([x, y]);
            weights.push(1.0 / n_elements as f64);
        }

        Ok(Self {
            config,
            element_positions: positions,
            element_weights: weights,
        })
    }

    /// Generate 2D source distribution
    #[must_use]
    pub fn generate_source_2d(&self, nx: usize, ny: usize, dx: f64, time: f64) -> Array2<f64> {
        let mut source = Array2::zeros((nx, ny));
        let omega = 2.0 * PI * self.config.frequency;

        // Focus is at the center of curvature
        let focus = self.config.center;

        // Calculate delays for focusing
        let speed_of_sound = SOUND_SPEED_WATER;
        let delays: Vec<f64> = self
            .element_positions
            .iter()
            .map(|&pos| {
                let distance = ((focus[0] - pos[0]).powi(2) + (focus[1] - pos[1]).powi(2)).sqrt();
                distance / speed_of_sound
            })
            .collect();

        // Generate source field
        Zip::indexed(&mut source).for_each(|(ix, iy), val| {
            let x = ix as f64 * dx;
            let y = iy as f64 * dx;

            let mut pressure = 0.0;
            for (i, &pos) in self.element_positions.iter().enumerate() {
                let r = ((x - pos[0]).powi(2) + (y - pos[1]).powi(2)).sqrt();

                if r > 0.0 {
                    // Phase with focusing delay
                    let phase = omega * (time - delays[i]);

                    // 2D Green's function (Hankel function calculation)
                    let element_pressure =
                        self.config.amplitude * self.element_weights[i] * phase.sin() / r.sqrt();

                    pressure += element_pressure;
                }
            }

            *val = pressure;
        });

        source
    }

    /// Extend 2D source to 3D (uniform in z-direction)
    pub fn generate_source_3d(&self, grid: &Grid, time: f64) -> Array3<f64> {
        let source_2d = self.generate_source_2d(grid.nx, grid.ny, grid.dx, time);
        let mut source_3d = Array3::zeros((grid.nx, grid.ny, grid.nz));

        // Copy 2D pattern to all z-slices
        for iz in 0..grid.nz {
            source_3d.slice_mut(s![.., .., iz]).assign(&source_2d);
        }

        source_3d
    }
}
