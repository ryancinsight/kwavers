//! Free functions for acoustic wave stability and nonlinearity computations.

use kwavers_domain::grid::Grid;
use kwavers_domain::medium::Medium;

use super::spatial_order::AcousticSpatialOrder;
use kwavers_core::constants::numerical::TWO_PI;

// Coefficient relating power-law absorption to acoustic diffusivity for soft tissues.
// Formula: δ ≈ 2αc³/(ω²). Reference: Szabo (1995) "Time domain wave equations for lossy media" Eq. 14.
const POWER_LAW_ABSORPTION_TO_DIFFUSIVITY_FACTOR: f64 = 2.0;

/// Computes acoustic diffusivity from power-law absorption.
///
/// Valid for biological soft tissues. For the complete viscosity/thermal formula see module docs.
///
/// # Physics
///
/// δ ≈ 2αc³/(ω²) where α is absorption coefficient, c sound speed, ω angular frequency.
///
/// Returns 0.0 at zero frequency (DC) to avoid division by zero.
#[must_use]
pub fn compute_diffusivity_from_power_law_absorption<M: Medium + ?Sized>(
    medium: &M,
    x: f64,
    y: f64,
    z: f64,
    frequency: f64,
    grid: &Grid,
) -> f64 {
    if frequency == 0.0 {
        return 0.0;
    }

    let alpha = kwavers_domain::medium::AcousticProperties::absorption_coefficient(
        medium, x, y, z, grid, frequency,
    );
    let (i, j, k) = kwavers_domain::medium::continuous_to_discrete(x, y, z, grid);
    let c = medium.sound_speed(i, j, k);
    let omega = TWO_PI * frequency;

    POWER_LAW_ABSORPTION_TO_DIFFUSIVITY_FACTOR * alpha * c.powi(3) / (omega * omega)
}

/// Compute the maximum stable time step for acoustic wave propagation (CFL condition).
///
/// dt_max = CFL_limit(order) · min(dx, dy, dz) / c_max
#[must_use]
pub fn compute_max_stable_timestep(
    grid: &Grid,
    max_sound_speed: f64,
    spatial_order: AcousticSpatialOrder,
) -> f64 {
    let min_dx = grid.dx.min(grid.dy).min(grid.dz);
    let cfl_limit = spatial_order.cfl_limit();
    cfl_limit * min_dx / max_sound_speed
}

/// Compute nonlinearity coefficient β = 1 + B/(2A) for the given medium and position.
#[must_use]
pub fn compute_nonlinearity_coefficient<M: Medium + ?Sized>(
    medium: &M,
    x: f64,
    y: f64,
    z: f64,
    grid: &Grid,
) -> f64 {
    // nonlinearity_parameter returns B/A; β = 1 + B/(2A)
    let b_over_a =
        kwavers_domain::medium::AcousticProperties::nonlinearity_parameter(medium, x, y, z, grid);
    1.0 + b_over_a / 2.0
}
