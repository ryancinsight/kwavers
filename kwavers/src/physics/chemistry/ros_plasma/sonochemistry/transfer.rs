//! Bubble → liquid ROS transfer and collapse energy estimation
//!
//! ## Theorem: Collapse Energy Conversion
//!
//! The total thermal energy of the gas interior at peak compression is
//! E_thermal = (3/2) N k_B T (equipartition for monatomic ideal gas).
//!
//! Only a fraction η ≈ 0.01 (1%) is converted to chemical dissociation
//! products. This efficiency is well-established in the sonochemistry
//! literature.
//!
//! **Reference**: Suslick, K. S. (1999). "Sonochemistry." In *Kirk-Othmer
//! Encyclopedia of Chemical Technology*. Wiley.

use super::model::BubbleState;
use crate::core::constants::fundamental::BOLTZMANN;

/// Fraction of collapse thermal energy converted to radical chemistry.
///
/// Reference: Suslick (1999), typical value for single-bubble sonoluminescence.
const COLLAPSE_ENERGY_CONVERSION_EFFICIENCY: f64 = 0.01;

/// Estimate energy deposited during bubble collapse
///
/// Uses kinetic theory: E = (3/2) N k_B T × η
///
/// where η = [`COLLAPSE_ENERGY_CONVERSION_EFFICIENCY`] ≈ 1%.
pub fn estimate_collapse_energy(state: &BubbleState) -> f64 {
    let n_total = state.n_gas + state.n_vapor;

    // Average thermal energy per molecule (3/2 kT for monatomic)
    let thermal_energy = 1.5 * n_total * BOLTZMANN * state.temperature;

    thermal_energy * COLLAPSE_ENERGY_CONVERSION_EFFICIENCY
}
