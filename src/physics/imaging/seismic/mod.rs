//! Seismic imaging methods
//!
//! # Literature References
//!
//! 1. **Virieux, J., & Operto, S. (2009)**. "An overview of full-waveform inversion
//!    in exploration geophysics." *Geophysics*, 74(6), WCC1-WCC26.
//!
//! 2. **Baysal, E., et al. (1983)**. "Reverse time migration." *Geophysics*,
//!    48(11), 1514-1524.

use crate::grid::Grid;
use ndarray::Array3;

/// Seismic imaging method types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeismicMethod {
    /// Full waveform inversion
    FullWaveformInversion,
    /// Reverse time migration
    ReverseTimeMigration,
    /// Kirchhoff migration
    KirchhoffMigration,
}

/// Seismic imaging configuration
#[derive(Debug, Clone)]
pub struct SeismicConfig {
    /// Imaging method
    pub method: SeismicMethod,
    /// Source frequency (Hz)
    pub frequency: f64,
    /// Maximum offset (m)
    pub max_offset: f64,
    /// Migration aperture angle (radians)
    pub aperture_angle: f64,
}

impl Default for SeismicConfig {
    fn default() -> Self {
        Self {
            method: SeismicMethod::ReverseTimeMigration,
            frequency: 30.0,
            max_offset: 3000.0,
            aperture_angle: 60.0 * std::f64::consts::PI / 180.0,
        }
    }
}

/// Apply imaging condition for RTM
///
/// Cross-correlation imaging condition: I(x) = Σₜ S(x,t) * R(x,t)
/// where S is source wavefield and R is receiver wavefield
#[must_use]
pub fn apply_imaging_condition(
    source_wavefield: &Array3<f64>,
    receiver_wavefield: &Array3<f64>,
) -> Array3<f64> {
    source_wavefield * receiver_wavefield
}

/// Compute migration weight based on illumination
#[must_use]
pub fn compute_illumination_compensation(
    source_wavefield: &Array3<f64>,
    epsilon: f64,
) -> Array3<f64> {
    let illumination = source_wavefield.mapv(|x| x * x);
    illumination.mapv(|x| 1.0 / (x + epsilon))
}

/// Apply Laplacian filter for artifact removal
pub fn apply_laplacian_filter(image: &Array3<f64>, grid: &Grid) -> Array3<f64> {
    let (nx, ny, nz) = image.dim();
    let mut filtered = Array3::zeros((nx, ny, nz));

    for i in 1..nx - 1 {
        for j in 1..ny - 1 {
            for k in 1..nz - 1 {
                let laplacian = (image[[i + 1, j, k]] - 2.0 * image[[i, j, k]]
                    + image[[i - 1, j, k]])
                    / (grid.dx * grid.dx)
                    + (image[[i, j + 1, k]] - 2.0 * image[[i, j, k]] + image[[i, j - 1, k]])
                        / (grid.dy * grid.dy)
                    + (image[[i, j, k + 1]] - 2.0 * image[[i, j, k]] + image[[i, j, k - 1]])
                        / (grid.dz * grid.dz);

                filtered[[i, j, k]] = -laplacian; // Negative for sharpening
            }
        }
    }

    filtered
}
