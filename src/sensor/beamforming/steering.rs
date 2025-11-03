//! Steering vector calculations for beamforming
//!
//! ## Mathematical Foundation
//! **Plane Wave**: a(θ,φ) = exp(j k r · û) where k = 2π/λ, û is unit direction vector
//! **Spherical Wave**: a(r) = exp(j k |r - r₀|) / |r - r₀| for near-field sources
//! **Focused Beam**: Combines phase delays for beam focusing at specific point

use ndarray::Array1;
use crate::error::KwaversResult;

/// Steering vector calculation methods
#[derive(Debug, Clone)]
pub enum SteeringVectorMethod {
    /// Far-field plane wave assumption: a(θ,φ) = exp(j k r · û)
    PlaneWave,
    /// Near-field spherical wave: a(r) = exp(j k |r - r₀|) / |r - r₀|
    SphericalWave { source_position: [f64; 3] },
    /// Focused beam at specific point
    Focused { focal_point: [f64; 3] },
}

/// Steering vector computation for array processing
#[derive(Debug)]
pub struct SteeringVector;

impl SteeringVector {
    /// Compute steering vector for given direction and sensor positions
    /// Returns complex-valued steering vector as `Array1<Complex<f64>>`
    pub fn compute(
        method: &SteeringVectorMethod,
        direction: [f64; 3],
        frequency: f64,
        sensor_positions: &[[f64; 3]],
        speed_of_sound: f64,
    ) -> KwaversResult<Array1<num_complex::Complex<f64>>> {
        use num_complex::Complex;

        let wavenumber = 2.0 * std::f64::consts::PI * frequency / speed_of_sound;
        let num_sensors = sensor_positions.len();

        let mut steering_vector = Array1::zeros(num_sensors);

        match method {
            SteeringVectorMethod::PlaneWave => {
                // Plane wave steering: a_i = exp(j k · r_i · û)
                // where û is the unit direction vector
                let direction_unit = Self::normalize_vector(direction);

                for (i, &pos) in sensor_positions.iter().enumerate() {
                    let phase = wavenumber * Self::dot_product(pos, direction_unit);
                    steering_vector[i] = Complex::new(0.0, phase).exp();
                }
            }

            SteeringVectorMethod::SphericalWave { source_position } => {
                // Spherical wave steering: a_i = exp(j k |r_i - r₀|) / |r_i - r₀|
                // where r₀ is the source position
                for (i, &pos) in sensor_positions.iter().enumerate() {
                    let distance = Self::euclidean_distance(pos, *source_position);
                    if distance.abs() < 1e-12 {
                        return Err(crate::error::KwaversError::Numerical(
                            crate::error::NumericalError::InvalidOperation(
                                format!("Sensor at source position (distance = {:.2e}) - spherical wave steering undefined", distance)
                            )
                        ));
                    }
                    let phase = wavenumber * distance;
                    let amplitude = 1.0 / distance; // Spherical spreading
                    steering_vector[i] = Complex::new(0.0, phase).exp() * amplitude;
                }
            }

            SteeringVectorMethod::Focused { focal_point } => {
                // Focused beam: phase delays to focus at focal_point
                for (i, &pos) in sensor_positions.iter().enumerate() {
                    let distance_to_focus = Self::euclidean_distance(pos, *focal_point);
                    let phase = wavenumber * distance_to_focus;
                    steering_vector[i] = Complex::new(0.0, phase).exp();
                }
            }
        }

        Ok(steering_vector)
    }

    /// Compute real-valued steering vector (phase-only for delay-and-sum)
    pub fn compute_real(
        method: &SteeringVectorMethod,
        direction: [f64; 3],
        frequency: f64,
        sensor_positions: &[[f64; 3]],
        speed_of_sound: f64,
    ) -> KwaversResult<Array1<f64>> {
        let complex_steering = Self::compute(method, direction, frequency, sensor_positions, speed_of_sound)?;
        Ok(complex_steering.mapv(|c| c.re)) // Take real part for delay-and-sum compatibility
    }

    /// Normalize 3D vector to unit length
    fn normalize_vector(v: [f64; 3]) -> [f64; 3] {
        let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if norm > 0.0 {
            [v[0] / norm, v[1] / norm, v[2] / norm]
        } else {
            [0.0, 0.0, 1.0] // Default to z-direction if zero vector
        }
    }

    /// Compute dot product of two 3D vectors
    fn dot_product(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    /// Compute Euclidean distance between two 3D points
    fn euclidean_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Compute broadside steering vector (perpendicular to array axis)
    #[must_use]
    pub fn broadside(sensor_positions: &[[f64; 3]], frequency: f64, speed_of_sound: f64) -> Array1<f64> {
        // Broadside: direction perpendicular to array (typically [0, 0, 1] or [0, 1, 0])
        Self::compute_real(
            &SteeringVectorMethod::PlaneWave,
            [0.0, 0.0, 1.0], // z-direction
            frequency,
            sensor_positions,
            speed_of_sound,
        ).unwrap_or_else(|_| Array1::ones(sensor_positions.len()))
    }

    /// Compute endfire steering vector (along array axis)
    #[must_use]
    pub fn endfire(sensor_positions: &[[f64; 3]], frequency: f64, speed_of_sound: f64) -> Array1<f64> {
        // Endfire: direction along array axis (typically [1, 0, 0])
        Self::compute_real(
            &SteeringVectorMethod::PlaneWave,
            [1.0, 0.0, 0.0], // x-direction
            frequency,
            sensor_positions,
            speed_of_sound,
        ).unwrap_or_else(|_| Array1::ones(sensor_positions.len()))
    }
}
