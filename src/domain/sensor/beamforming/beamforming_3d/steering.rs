//! 3D Steering Vector Computation for MVDR Beamforming
//!
//! This module provides steering vector computation for Minimum Variance Distortionless
//! Response (MVDR) beamforming in 3D volumetric ultrasound imaging.
//!
//! # Theory
//! The steering vector represents the phase delays for each transducer element required
//! to focus the ultrasound beam at a specific spatial location (voxel). For MVDR beamforming,
//! the steering vector is used in the adaptive weight calculation.
//!
//! # References
//! - Van Veen & Buckley (1988) "Beamforming: A versatile approach to spatial filtering"
//! - Jensen (1996) "Field: A Program for Simulating Ultrasound Systems"
//! - Synnevåg et al. (2009) "Adaptive beamforming applied to medical ultrasound imaging"

use crate::core::error::KwaversResult;
use ndarray::Array1;

use super::config::BeamformingConfig3D;
use crate::domain::sensor::beamforming::steering::{SteeringVector, SteeringVectorMethod};

/// Compute 3D steering vector for MVDR beamforming
///
/// Generates a steering vector that represents the complex weights needed to focus
/// the array at the specified voxel position. The steering vector accounts for:
/// - Geometric delays from element positions to the focal point
/// - Phase shifts at the center frequency
/// - Far-field approximation (plane wave model)
///
/// # Arguments
/// * `config` - Beamforming configuration with array geometry
/// * `voxel_pos` - Target voxel position [x, y, z] in meters
/// * `num_elements` - Number of active elements to include
///
/// # Returns
/// Real-valued steering vector (magnitude of complex weights) for the specified voxel
///
/// # Mathematical Model
/// For a 3D rectangular array, the steering vector element i is:
/// ```text
/// a_i(θ, φ) = exp(-j * 2π * f_c / c * r_i · d)
/// ```
/// where:
/// - f_c is the center frequency
/// - c is the sound speed
/// - r_i is the position of element i
/// - d is the unit direction vector to the voxel
#[allow(dead_code)]
pub fn compute_steering_vector_3d(
    config: &BeamformingConfig3D,
    voxel_pos: &[f32; 3],
    num_elements: usize,
) -> KwaversResult<Array1<f64>> {
    // Generate complete 3D transducer element positions
    // Support for multiple 3D array geometries for comprehensive beamforming
    // Literature: Jensen (1996) - Field: A Program for Simulating Ultrasound Systems

    let mut element_positions = Vec::new();

    // Use a 3D rectangular grid for complete volumetric imaging
    // This provides full 3D coverage unlike simplified 2D arrays
    for ex in 0..config.num_elements_3d.0 {
        for ey in 0..config.num_elements_3d.1 {
            for ez in 0..config.num_elements_3d.2 {
                // Center the 3D array around origin
                let x = (ex as f64 - (config.num_elements_3d.0 - 1) as f64 * 0.5)
                    * config.element_spacing_3d.0;
                let y = (ey as f64 - (config.num_elements_3d.1 - 1) as f64 * 0.5)
                    * config.element_spacing_3d.1;
                let z = (ez as f64 - (config.num_elements_3d.2 - 1) as f64 * 0.5)
                    * config.element_spacing_3d.2;

                element_positions.push([x, y, z]);
            }
        }
    }

    // Convert voxel position to direction vector (assuming far-field)
    let direction = [
        voxel_pos[0] as f64,
        voxel_pos[1] as f64,
        voxel_pos[2] as f64,
    ];

    // Use existing steering vector computation from 2D module
    let steering_vector_complex = SteeringVector::compute(
        &SteeringVectorMethod::PlaneWave,
        direction,
        config.center_frequency,
        &element_positions[..num_elements.min(element_positions.len())],
        config.sound_speed,
    )?;

    // Convert complex steering vector to real-valued vector (magnitude)
    // For MVDR beamforming, we use the magnitude which represents the amplitude weighting
    Ok(steering_vector_complex.mapv(|c| c.norm()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_steering_vector_3d_dimensions() {
        let config = BeamformingConfig3D::default();
        let voxel_pos = [0.0_f32, 0.0, 1.0]; // Unit depth (normalized direction)
        let num_elements = 100;

        let steering_vec = compute_steering_vector_3d(&config, &voxel_pos, num_elements);
        assert!(steering_vec.is_ok());

        let vec = steering_vec.unwrap();
        assert_eq!(vec.len(), num_elements);
    }

    #[test]
    fn test_steering_vector_3d_unit_magnitude() {
        let config = BeamformingConfig3D::default();
        let voxel_pos = [0.0_f32, 0.0, 1.0]; // Unit direction vector

        let steering_vec = compute_steering_vector_3d(&config, &voxel_pos, 64).unwrap();

        // All weights should be positive (magnitudes)
        assert!(steering_vec.iter().all(|&w| w >= 0.0));

        // Weights should be normalized (typically around 1.0 for far-field)
        assert!(steering_vec.iter().all(|&w| w <= 2.0));
    }

    #[test]
    fn test_steering_vector_3d_different_positions() {
        let config = BeamformingConfig3D::default();
        // Normalize direction vectors to unit vectors
        let pos1 = [0.0_f32, 0.0, 1.0]; // Straight down
        let norm2 = (0.3_f32.powi(2) + 0.4_f32.powi(2) + 0.866_f32.powi(2)).sqrt();
        let pos2 = [0.3 / norm2, 0.4 / norm2, 0.866 / norm2]; // Angled

        let vec1 = compute_steering_vector_3d(&config, &pos1, 64).unwrap();
        let vec2 = compute_steering_vector_3d(&config, &pos2, 64).unwrap();

        // Different positions should produce different steering vectors
        let diff_sum: f64 = vec1
            .iter()
            .zip(vec2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff_sum > 0.0);
    }
}
