//! Analytical solutions for validation
//!
//! Implements exact analytical solutions for acoustic wave propagation
//! to validate numerical methods against known results.
//!
//! References:
//! - Blackstock, D.T. (2000) "Fundamentals of Physical Acoustics"
//! - Pierce, A.D. (1989) "Acoustics: An Introduction to Its Physical Principles"

use crate::domain::grid::Grid;
use ndarray::{Array3, Zip};
use std::f64::consts::PI;

/// Analytical solution for plane wave propagation
#[derive(Debug)]
pub struct PlaneWaveSolution {
    /// Wave frequency (Hz)
    pub frequency: f64,
    /// Wave speed (m/s)
    pub sound_speed: f64,
    /// Wave amplitude (Pa)
    pub amplitude: f64,
    /// Propagation direction (normalized)
    pub direction: [f64; 3],
}

impl PlaneWaveSolution {
    /// Create a new plane wave solution
    pub fn new(frequency: f64, sound_speed: f64, amplitude: f64, direction: [f64; 3]) -> Self {
        // Normalize direction
        let norm = (direction[0].powi(2) + direction[1].powi(2) + direction[2].powi(2)).sqrt();
        Self {
            frequency,
            sound_speed,
            amplitude,
            direction: [
                direction[0] / norm,
                direction[1] / norm,
                direction[2] / norm,
            ],
        }
    }

    /// Evaluate pressure at given position and time
    pub fn pressure(&self, x: f64, y: f64, z: f64, t: f64) -> f64 {
        let k = 2.0 * PI * self.frequency / self.sound_speed; // Wavenumber
        let omega = 2.0 * PI * self.frequency; // Angular frequency

        // Phase: k·r - ωt
        let phase =
            k * (self.direction[0] * x + self.direction[1] * y + self.direction[2] * z) - omega * t;

        self.amplitude * phase.cos()
    }

    /// Evaluate velocity at given position and time
    pub fn velocity(&self, x: f64, y: f64, z: f64, t: f64) -> [f64; 3] {
        let k = 2.0 * PI * self.frequency / self.sound_speed;
        let omega = 2.0 * PI * self.frequency;
        let impedance = 1500.0 * 1000.0; // ρc for water

        let phase =
            k * (self.direction[0] * x + self.direction[1] * y + self.direction[2] * z) - omega * t;

        let v_magnitude = self.amplitude / impedance * phase.cos();

        [
            v_magnitude * self.direction[0],
            v_magnitude * self.direction[1],
            v_magnitude * self.direction[2],
        ]
    }

    /// Generate pressure field on grid
    pub fn pressure_field(&self, grid: &Grid, t: f64) -> Array3<f64> {
        let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));

        Zip::indexed(&mut field).for_each(|(i, j, k), p| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;
            *p = self.pressure(x, y, z, t);
        });

        field
    }
}

/// Analytical solution for point source (Green's function)
#[derive(Debug)]
pub struct PointSourceSolution {
    /// Source frequency (Hz)
    pub frequency: f64,
    /// Wave speed (m/s)
    pub sound_speed: f64,
    /// Source strength (m³/s)
    pub strength: f64,
    /// Source position
    pub position: [f64; 3],
}

impl PointSourceSolution {
    /// Create a new point source solution
    pub fn new(frequency: f64, sound_speed: f64, strength: f64, position: [f64; 3]) -> Self {
        Self {
            frequency,
            sound_speed,
            strength,
            position,
        }
    }

    /// Evaluate pressure at given position and time (3D Green's function)
    pub fn pressure(&self, x: f64, y: f64, z: f64, t: f64) -> f64 {
        // Distance from source
        let r = ((x - self.position[0]).powi(2)
            + (y - self.position[1]).powi(2)
            + (z - self.position[2]).powi(2))
        .sqrt();

        if r < 1e-10 {
            return 0.0; // Avoid singularity
        }

        let k = 2.0 * PI * self.frequency / self.sound_speed;
        let omega = 2.0 * PI * self.frequency;
        let density = 1000.0; // Water density

        // 3D Green's function: G = (1/4πr) * exp(i(kr - ωt))
        // Pressure: p = iωρQ * G
        let phase = k * r - omega * t;

        omega * density * self.strength / (4.0 * PI * r) * phase.sin()
    }

    /// Generate pressure field on grid
    pub fn pressure_field(&self, grid: &Grid, t: f64) -> Array3<f64> {
        let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));

        Zip::indexed(&mut field).for_each(|(i, j, k), p| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;
            *p = self.pressure(x, y, z, t);
        });

        field
    }
}

/// Analytical solution for standing wave in rectangular cavity
#[derive(Debug)]
pub struct StandingWaveSolution {
    /// Cavity dimensions (m)
    pub dimensions: [f64; 3],
    /// Mode numbers
    pub modes: [usize; 3],
    /// Wave speed (m/s)
    pub sound_speed: f64,
    /// Amplitude (Pa)
    pub amplitude: f64,
}

impl StandingWaveSolution {
    /// Create a new standing wave solution
    pub fn new(dimensions: [f64; 3], modes: [usize; 3], sound_speed: f64, amplitude: f64) -> Self {
        Self {
            dimensions,
            modes,
            sound_speed,
            amplitude,
        }
    }

    /// Calculate eigenfrequency for the mode
    pub fn eigenfrequency(&self) -> f64 {
        let kx = self.modes[0] as f64 * PI / self.dimensions[0];
        let ky = self.modes[1] as f64 * PI / self.dimensions[1];
        let kz = self.modes[2] as f64 * PI / self.dimensions[2];

        self.sound_speed / (2.0 * PI) * (kx.powi(2) + ky.powi(2) + kz.powi(2)).sqrt()
    }

    /// Evaluate pressure at given position and time
    pub fn pressure(&self, x: f64, y: f64, z: f64, t: f64) -> f64 {
        let kx = self.modes[0] as f64 * PI / self.dimensions[0];
        let ky = self.modes[1] as f64 * PI / self.dimensions[1];
        let kz = self.modes[2] as f64 * PI / self.dimensions[2];

        let omega = 2.0 * PI * self.eigenfrequency();

        self.amplitude * (kx * x).cos() * (ky * y).cos() * (kz * z).cos() * (omega * t).cos()
    }

    /// Generate pressure field on grid
    pub fn pressure_field(&self, grid: &Grid, t: f64) -> Array3<f64> {
        let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));

        Zip::indexed(&mut field).for_each(|(i, j, k), p| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;
            *p = self.pressure(x, y, z, t);
        });

        field
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_plane_wave_propagation() {
        let solution = PlaneWaveSolution::new(
            1000.0,          // 1 kHz
            1500.0,          // Water sound speed
            1e5,             // 100 kPa
            [1.0, 0.0, 0.0], // x-direction
        );

        // Test at origin, t=0
        let p0 = solution.pressure(0.0, 0.0, 0.0, 0.0);
        assert_relative_eq!(p0, 1e5, epsilon = 1e-10);

        // Test phase shift after quarter period
        let period = 1.0 / 1000.0;
        let p_quarter = solution.pressure(0.0, 0.0, 0.0, period / 4.0);
        assert_relative_eq!(p_quarter, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_point_source_decay() {
        let solution = PointSourceSolution::new(
            1000.0,          // 1 kHz
            1500.0,          // Water sound speed
            1e-6,            // Source strength
            [0.0, 0.0, 0.0], // Origin
        );

        // Test 1/r decay
        let p1 = solution.pressure(1.0, 0.0, 0.0, 0.0).abs();
        let p2 = solution.pressure(2.0, 0.0, 0.0, 0.0).abs();

        // Pressure should decay as 1/r
        assert_relative_eq!(p1 / p2, 2.0, epsilon = 0.01);
    }

    #[test]
    fn test_standing_wave_nodes() {
        let solution = StandingWaveSolution::new(
            [1.0, 1.0, 1.0], // 1m cube
            [1, 0, 0],       // First mode in x
            1500.0,          // Sound speed
            1e5,             // Amplitude
        );

        // Test pressure at antinode (maximum)
        let p_max = solution.pressure(0.0, 0.5, 0.5, 0.0);
        assert_relative_eq!(p_max, 1e5, epsilon = 1e-10);

        // Test pressure at node (zero)
        let p_node = solution.pressure(0.5, 0.5, 0.5, 0.0);
        assert_relative_eq!(p_node, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_eigenfrequency_calculation() {
        let solution = StandingWaveSolution::new(
            [1.0, 1.0, 1.0], // 1m cube
            [1, 1, 1],       // First diagonal mode
            1500.0,          // Sound speed
            1e5,             // Amplitude
        );

        let freq = solution.eigenfrequency();
        // Expected: c/(2π) * π√3 = c√3/2 ≈ 1299 Hz
        let expected = 1500.0 * (3.0_f64).sqrt() / 2.0;
        assert_relative_eq!(freq, expected, epsilon = 1.0);
    }
}
