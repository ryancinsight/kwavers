//! Nonlinear acoustics implementation for Spectral parity
//!
//! Implements the nonlinear term: ∂ρ/∂t = -∇·(ρ₀u) - ∇·(ρu)
//! where the second term captures nonlinear effects through B/A parameter
//!
//! References:
//! - Hamilton & Blackstock (1998) "Nonlinear Acoustics"
//! - Aanonsen et al. (1984) "Distortion and harmonic generation"

use crate::domain::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::Array3;

/// Default B/A parameter for water at 20°C
const BA_WATER: f64 = 5.0;

/// Default B/A parameter for soft tissue
const BA_TISSUE: f64 = 6.0;

/// Update pressure with nonlinear effects
pub fn update_pressure_with_nonlinearity(
    p: &mut Array3<f64>,
    div_u: &Array3<f64>,
    medium: &dyn Medium,
    grid: &Grid,
    dt: f64,
) -> KwaversResult<()> {
    // Nonlinear coefficient of acoustics
    // β = 1 + B/2A where B/A is the nonlinearity parameter

    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let (x, y, z) = grid.indices_to_coordinates(i, j, k);

                // Get medium properties
                let rho0 = crate::domain::medium::density_at(medium, x, y, z, grid);
                let c0 = crate::domain::medium::sound_speed_at(medium, x, y, z, grid);

                // Get B/A parameter (would be from medium in full implementation)
                let b_over_a = get_nonlinearity_parameter(medium, x, y, z, grid);
                let beta = 1.0 + b_over_a / 2.0;

                // Linear term
                let linear = -rho0 * c0 * c0 * div_u[[i, j, k]];

                // Nonlinear term (convective + parameter nonlinearity)
                // Includes the ρ = p/(c²) substitution
                let density_variation = p[[i, j, k]] / (c0 * c0);
                let nonlinear = -beta * density_variation * c0 * c0 * div_u[[i, j, k]];

                // Update pressure
                p[[i, j, k]] += dt * (linear + nonlinear);
            }
        }
    }

    Ok(())
}

/// Get B/A nonlinearity parameter for medium
/// Uses sound speed heuristic when medium doesn't provide direct B/A access
/// Per Hamilton & Blackstock (1998) Table 3.1: typical tissue values 6-11, water ~5
fn get_nonlinearity_parameter(medium: &dyn Medium, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
    // Sound speed-based heuristic provides reasonable B/A estimates
    // Future: Add Medium::nonlinearity_coefficient() trait method (Sprint 124+)

    let c = crate::domain::medium::sound_speed_at(medium, x, y, z, grid);

    if c < 1450.0 {
        // Likely fat/oil
        4.5
    } else if c < 1550.0 {
        // Likely water
        BA_WATER
    } else {
        // Likely tissue
        BA_TISSUE
    }
}

/// Compute cumulative nonlinearity for shock formation
pub fn compute_cumulative_nonlinearity(
    p: &Array3<f64>,
    medium: &dyn Medium,
    grid: &Grid,
    propagation_distance: f64,
) -> Array3<f64> {
    let mut sigma = Array3::zeros(p.dim());

    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let (x, y, z) = grid.indices_to_coordinates(i, j, k);

                let rho0 = crate::domain::medium::density_at(medium, x, y, z, grid);
                let c0 = crate::domain::medium::sound_speed_at(medium, x, y, z, grid);
                let b_over_a = get_nonlinearity_parameter(medium, x, y, z, grid);
                let beta = 1.0 + b_over_a / 2.0;

                // Cumulative nonlinearity parameter (Goldberg number)
                // σ = βkxp₀/ρ₀c₀³
                let k_wave = 2.0 * std::f64::consts::PI * 1e6 / c0; // Assume 1 MHz
                sigma[[i, j, k]] =
                    beta * k_wave * propagation_distance * p[[i, j, k]].abs() / (rho0 * c0.powi(3));
            }
        }
    }

    sigma
}

/// Check if shock formation is occurring
#[must_use]
pub fn is_shock_forming(sigma: &Array3<f64>) -> bool {
    // Shock forms when σ > 1
    sigma.iter().any(|&s| s > 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::medium::HomogeneousMedium;

    #[test]
    fn test_nonlinearity_parameter() {
        let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
        let medium = HomogeneousMedium::water(&grid);

        let b_over_a = get_nonlinearity_parameter(&medium, 0.0, 0.0, 0.0, &grid);

        // Water should have B/A ≈ 5
        assert!((b_over_a - BA_WATER).abs() < 1.0);
    }

    #[test]
    fn test_shock_detection() {
        let mut sigma = Array3::zeros((10, 10, 10));

        // No shock
        assert!(!is_shock_forming(&sigma));

        // Create shock condition
        sigma[[5, 5, 5]] = 1.5;
        assert!(is_shock_forming(&sigma));
    }
}
