//! Muscle fiber modeling for anisotropic biological tissues
//!
//! References:
//! - Blemker et al. (2005). "A 3D model of muscle reveals the causes of nonuniform strains"

use crate::Grid;
use ndarray::Array3;
use std::f64::consts::PI;

/// Fiber orientation in 3D space
#[derive(Debug, Clone, Default)]
pub struct FiberOrientation {
    /// Azimuthal angle (φ) in radians
    pub azimuth: f64,
    /// Polar angle (θ) in radians
    pub elevation: f64,
}

impl FiberOrientation {
    /// Create from angles
    pub fn from_angles(azimuth: f64, elevation: f64) -> Self {
        Self { azimuth, elevation }
    }

    /// Convert to unit vector
    pub fn to_vector(&self) -> [f64; 3] {
        let (sin_theta, cos_theta) = self.elevation.sin_cos();
        let (sin_phi, cos_phi) = self.azimuth.sin_cos();

        [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta]
    }

    /// Create from vector
    pub fn from_vector(v: &[f64; 3]) -> Self {
        let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if r < 1e-10 {
            return Self {
                azimuth: 0.0,
                elevation: 0.0,
            };
        }

        let elevation = (v[2] / r).acos();
        let azimuth = v[1].atan2(v[0]);

        Self { azimuth, elevation }
    }
}

/// Muscle fiber model for anisotropic wave propagation
#[derive(Debug)]
pub struct MuscleFiberModel {
    /// Fiber orientations at each grid point
    fiber_field: Array3<FiberOrientation>,
    /// Pennation angle (angle between fibers and force direction)
    pennation_angle: f64,
    /// Fiber volume fraction
    volume_fraction: f64,
}

impl MuscleFiberModel {
    /// Create uniform fiber model
    pub fn uniform(grid: &Grid, orientation: FiberOrientation, pennation_angle: f64) -> Self {
        let fiber_field = Array3::from_elem((grid.nx, grid.ny, grid.nz), orientation.clone());

        Self {
            fiber_field,
            pennation_angle,
            volume_fraction: 0.3, // Typical for muscle
        }
    }

    /// Create helical fiber arrangement (e.g., cardiac muscle)
    pub fn helical(grid: &Grid, pitch: f64, radius: f64) -> Self {
        let mut fiber_field = Array3::default((grid.nx, grid.ny, grid.nz));

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;

                    // Helical angle varies with radius
                    let r = ((x * x + y * y).sqrt() / radius).min(1.0);
                    let helix_angle = -PI / 3.0 + r * 2.0 * PI / 3.0; // -60° to +60°

                    let azimuth = y.atan2(x) + helix_angle;
                    let elevation = (z / pitch).atan();

                    fiber_field[[i, j, k]] = FiberOrientation::from_angles(azimuth, elevation);
                }
            }
        }

        Self {
            fiber_field,
            pennation_angle: 0.0,
            volume_fraction: 0.3,
        }
    }

    /// Get fiber orientation at specific location
    pub fn orientation_at(&self, i: usize, j: usize, k: usize) -> &FiberOrientation {
        &self.fiber_field[[i, j, k]]
    }

    /// Calculate local stiffness enhancement along fiber direction
    pub fn stiffness_enhancement(&self, along_fiber: bool) -> f64 {
        if along_fiber {
            // Fibers are stiffer along their length
            1.0 + 2.0 * self.volume_fraction
        } else {
            // Softer perpendicular to fibers
            1.0 - 0.5 * self.volume_fraction
        }
    }

    /// Apply pennation correction to wave speed
    pub fn pennation_correction(&self, wave_speed: f64) -> f64 {
        // Pennation reduces effective stiffness
        wave_speed * self.pennation_angle.cos()
    }

    /// Check if point is within fiber bundle
    pub fn is_fiber_region(&self, i: usize, j: usize, k: usize) -> bool {
        // Could implement more complex fiber bundle geometry
        i < self.fiber_field.shape()[0]
            && j < self.fiber_field.shape()[1]
            && k < self.fiber_field.shape()[2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fiber_orientation_conversion() {
        let orientation = FiberOrientation::from_angles(0.0, PI / 2.0);
        let vector = orientation.to_vector();

        assert!((vector[0] - 1.0).abs() < 1e-10);
        assert!(vector[1].abs() < 1e-10);
        assert!(vector[2].abs() < 1e-10);
    }

    #[test]
    fn test_fiber_orientation_roundtrip() {
        let original = FiberOrientation::from_angles(PI / 4.0, PI / 3.0);
        let vector = original.to_vector();
        let recovered = FiberOrientation::from_vector(&vector);

        assert!((original.azimuth - recovered.azimuth).abs() < 1e-10);
        assert!((original.elevation - recovered.elevation).abs() < 1e-10);
    }
}
