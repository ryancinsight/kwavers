//! Photoacoustic imaging physics
//!
//! # Literature References
//!
//! 1. **Wang, L. V., & Hu, S. (2012)**. "Photoacoustic tomography: in vivo imaging
//!    from organelles to organs." *Science*, 335(6075), 1458-1462.
//!
//! 2. **Xu, M., & Wang, L. V. (2006)**. "Photoacoustic imaging in biomedicine."
//!    *Review of Scientific Instruments*, 77(4), 041101.

use crate::{error::KwaversResult, grid::Grid, medium::Medium};
use ndarray::Array3;
use std::f64::consts::PI;

/// Photoacoustic imaging configuration
#[derive(Debug, Clone))]
pub struct PhotoacousticConfig {
    /// Optical wavelength (m)
    pub wavelength: f64,
    /// Pulse duration (s)
    pub pulse_duration: f64,
    /// Fluence (J/m²)
    pub fluence: f64,
    /// Grueneisen parameter
    pub grueneisen: f64,
}

impl Default for PhotoacousticConfig {
    fn default() -> Self {
        Self {
            wavelength: 800e-9,    // 800 nm (NIR)
            pulse_duration: 10e-9, // 10 ns
            fluence: 20e-3,        // 20 mJ/cm²
            grueneisen: 0.16,      // Typical for tissue
        }
    }
}

/// Photoacoustic initial pressure distribution
///
/// Implements p₀ = Γ * μₐ * Φ
/// where:
/// - Γ is the Grueneisen parameter
/// - μₐ is the optical absorption coefficient
/// - Φ is the optical fluence
pub fn compute_initial_pressure(
    optical_absorption: &Array3<f64>,
    fluence: &Array3<f64>,
    config: &PhotoacousticConfig,
) -> Array3<f64> {
    optical_absorption * fluence * config.grueneisen
}

/// Compute optical fluence distribution using diffusion approximation
///
/// Solves: -∇·(D∇Φ) + μₐΦ = S
/// where D = 1/(3(μₐ + μ'ₛ)) is the diffusion coefficient
pub fn compute_fluence_diffusion(
    absorption: &Array3<f64>,
    scattering: &Array3<f64>,
    source: &Array3<f64>,
    grid: &Grid,
) -> KwaversResult<Array3<f64>> {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let mut fluence = Array3::zeros((nx, ny, nz));

    // Compute diffusion coefficient
    let diffusion_coeff = 1.0 / (3.0 * (absorption + scattering));

    // Iterative solver for diffusion equation
    const MAX_ITERATIONS: usize = 1000;
    const TOLERANCE: f64 = 1e-6;

    for iteration in 0..MAX_ITERATIONS {
        let fluence_prev = fluence.clone();

        // Update fluence using finite differences
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let d = diffusion_coeff[[i, j, k];
                    let mu_a = absorption[[i, j, k];

                    // Laplacian using central differences
                    let laplacian = (fluence_prev[[i + 1, j, k] - 2.0 * fluence_prev[[i, j, k]
                        + fluence_prev[[i - 1, j, k])
                        / (grid.dx * grid.dx)
                        + (fluence_prev[[i, j + 1, k] - 2.0 * fluence_prev[[i, j, k]
                            + fluence_prev[[i, j - 1, k])
                            / (grid.dy * grid.dy)
                        + (fluence_prev[[i, j, k + 1] - 2.0 * fluence_prev[[i, j, k]
                            + fluence_prev[[i, j, k - 1])
                            / (grid.dz * grid.dz);

                    // Update equation: Φ = (S + D∇²Φ) / μₐ
                    fluence[[i, j, k] = (source[[i, j, k] + d * laplacian) / (mu_a + 1e-10);
                }
            }
        }

        // Check convergence
        let error = (&fluence - &fluence_prev).mapv(f64::abs).sum();
        if error < TOLERANCE {
            break;
        }

        if iteration == MAX_ITERATIONS - 1 {
            log::warn!("Fluence computation did not converge");
        }
    }

    Ok(fluence)
}

/// Compute sensitivity matrix for photoacoustic tomography
///
/// The sensitivity matrix relates the initial pressure to the detected signals
pub fn compute_sensitivity_matrix(
    grid: &Grid,
    detector_positions: &[[f64; 3],
    medium: &dyn Medium,
) -> KwaversResult<Array3<f64>> {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let mut sensitivity = Array3::zeros((nx, ny, nz));

    for detector_pos in detector_positions {
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let (x, y, z) =
                        crate::grid::coordinates::CoordinateSystem::indices_to_position(
                            grid, i, j, k,
                        )
                        .unwrap();

                    // Distance from voxel to detector
                    let dx = x - detector_pos[0];
                    let dy = y - detector_pos[1];
                    let dz = z - detector_pos[2];
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                    // Sound speed at voxel
                    let c = medium.sound_speed(x, y, z, grid);

                    // Sensitivity decreases with distance (spherical spreading)
                    // and depends on acoustic properties
                    sensitivity[[i, j, k] += 1.0 / (4.0 * PI * distance.max(grid.dx));
                }
            }
        }
    }

    Ok(sensitivity)
}
