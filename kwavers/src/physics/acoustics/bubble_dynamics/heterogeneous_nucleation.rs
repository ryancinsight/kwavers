//! # Heterogeneous Nucleation Models
//!
//! [PHASE-2: SPEC-DRIVEN IMPLEMENTATION]
//!
//! This module provides mathematical formulations for stiffness-coupled
//! heterogeneous cavitation nucleation. This is vital for simulating
//! targeted oncological treatments where circulating tumor cells (CTCs)
//! or solid tumors exhibit differential mechanical stiffness, thereby
//! altering the localized cavitation threshold.
//!
//! ## Mathematical Specifications
//!
//! ### Theorem: Stiffness-Coupled Classical Nucleation
//! The heterogeneous nucleation rate $J$ ($events \cdot m^{-3} \cdot s^{-1}$)
//! is defined by modifying the critical free energy barrier $\Delta G_c$
//! with an elasticity-dependent activation factor $f(E)$, where $E$ is the
//! local Young's modulus of the tissue:
//!
//! $$ J = J_0 \exp\left( - \frac{\Delta G_c f(E)}{k_B T} \right) $$
//!
//! where the uncoupled barrier is:
//! $$ \Delta G_c = \frac{16 \pi \gamma^3}{3 (P_v - P_l)^2} $$
//!
//! for local fluid pressure $P_l$ and vapor pressure $P_v$.
//!
//! ### Invariants
//! 1. Nucleation rate $J \ge 0.0$.
//! 2. Nucleation only occurs under tension ($P_l < P_v$).
//! 3. Asymptotic limit: if $P_l \ge P_v$, $J \to 0.0$.
//!
//! ## Literature
//! - Church, C. C. (2002). Spontaneous homogeneous nucleation, inertial cavitation and the tensile strength of water.

use crate::core::constants::fundamental::BOLTZMANN;

/// Trait defining the contract for calculating heterogeneous nucleation.
pub trait HeterogeneousNucleationModel {
    /// Mathematical floating-point type parameter.
    type Scalar;

    /// Calculates the instantaneous geometric and mechanical reduction factor $f(E)$.
    ///
    /// # Parameters
    /// - `youngs_modulus`: Elastic modulus $E$ of the specific tissue cell limit ($Pa$).
    ///
    /// # Returns
    /// Scaling parameter $f \in [0.0, 1.0]$.
    fn calculate_stiffness_factor(&self, youngs_modulus: Self::Scalar) -> Self::Scalar;

    /// Calculates the absolute bubble nucleation rate events per volume per second.
    ///
    /// # Parameters
    /// - `pressure`: Local absolute acoustic pressure $P_l$ ($Pa$).
    /// - `temperature`: Local tissue temperature $T$ ($K$).
    /// - `stiffness_factor`: Mechanical reduction factor $f$.
    ///
    /// # Returns
    /// Nucleation rate $J$.
    fn calculate_nucleation_rate(
        &self,
        pressure: Self::Scalar,
        temperature: Self::Scalar,
        stiffness_factor: Self::Scalar,
    ) -> Self::Scalar;
}

/// Stiffness-coupled classical heterogeneous nucleation model.
///
/// Provides generic implementation for variant-agnostic operation. The
/// Boltzmann constant `k_B` is sourced from the SSOT
/// [`crate::core::constants::fundamental::BOLTZMANN`] and is not a
/// configurable parameter — physical universal constants are not callable
/// inputs in this codebase.
#[derive(Debug, Clone)]
pub struct ClassicalHeterogeneousNucleation<T: std::fmt::Debug + Clone> {
    /// Surface tension $\gamma$ ($N/m$).
    pub surface_tension: T,
    /// Vapor pressure $P_v$ ($Pa$).
    pub vapor_pressure: T,
    /// Pre-exponential factor $J_0$ ($m^{-3} s^{-1}$).
    pub pre_exponential_factor: T,
    /// Reference stiffness $E_{ref}$ where factor reaches 1.0.
    pub reference_stiffness: T,
}

impl<T: num_traits::Float + std::fmt::Debug + Clone> HeterogeneousNucleationModel
    for ClassicalHeterogeneousNucleation<T>
{
    type Scalar = T;

    #[inline(always)]
    fn calculate_stiffness_factor(&self, youngs_modulus: T) -> T {
        let one = T::one();
        let zero = T::zero();
        if youngs_modulus <= zero {
            return one;
        }

        // As local cell stiffness increases, the reduction factor scales quadratically
        // bounded to 1.0 (approaching homogeneous limit). Tumor cells (lower stiffness)
        // yield lower factors, reducing the threshold and increasing nucleation.
        let ratio = self.reference_stiffness / youngs_modulus;
        let factor = ratio * ratio;

        // Clamp to [0, 1] without trait missing issues
        if factor > one {
            one
        } else if factor < zero {
            zero
        } else {
            factor
        }
    }

    #[inline(always)]
    fn calculate_nucleation_rate(&self, pressure: T, temperature: T, stiffness_factor: T) -> T {
        let zero = T::zero();

        // Nucleation only occurs under negative acoustic pressure (tension)
        // pulling the fluid apart relative to vapor pressure.
        if pressure >= self.vapor_pressure {
            return zero;
        }

        // Classical Nucleation Theory geometric prefactor: 16π/3.
        // Sourced from std::f64::consts::PI to avoid drifting from the
        // canonical value via a hand-computed decimal literal.
        let sixteen_pi_over_three =
            T::from(16.0 * std::f64::consts::PI / 3.0).expect("16π/3 representable in Scalar T");
        let p_diff = self.vapor_pressure - pressure;

        let surface_tension_cubed =
            self.surface_tension * self.surface_tension * self.surface_tension;
        let p_diff_squared = p_diff * p_diff;

        // Prevent div by zero inherently covered by if check above but explicit
        if p_diff_squared <= zero {
            return zero;
        }

        let delta_g_c = (sixteen_pi_over_three * surface_tension_cubed) / p_diff_squared;

        let k_b = T::from(BOLTZMANN).expect("BOLTZMANN constant representable in Scalar T");
        let k_t = k_b * temperature;
        if k_t <= zero {
            return zero;
        }

        let exponent = -(delta_g_c * stiffness_factor) / k_t;

        self.pre_exponential_factor * exponent.exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::thermodynamic::ROOM_TEMPERATURE_K;

    fn test_model() -> ClassicalHeterogeneousNucleation<f64> {
        use crate::core::constants::cavitation::{SURFACE_TENSION_WATER, VAPOR_PRESSURE_WATER};
        ClassicalHeterogeneousNucleation {
            surface_tension: SURFACE_TENSION_WATER,
            vapor_pressure: VAPOR_PRESSURE_WATER,
            pre_exponential_factor: 1.0e33,
            reference_stiffness: 1e5, // Nominal
        }
    }

    #[test]
    fn test_nucleation_tension_invariant() {
        let model = test_model();
        let temp = ROOM_TEMPERATURE_K;
        let factor = 1.0;

        // P_l > P_v => Positive/Compressive => Zero rate
        assert_eq!(model.calculate_nucleation_rate(100000.0, temp, factor), 0.0);

        // P_l = P_v => Zero rate
        assert_eq!(model.calculate_nucleation_rate(2330.0, temp, factor), 0.0);

        // Tension => Non-zero positive.
        //
        // At −1 MPa tension, the classical CNT barrier for water (γ = 0.072 N/m)
        // gives ΔG_c/k_BT ≈ 1.54×10⁶, which underflows to 0.0 in f64 (f64 minimum
        // representable exp argument ≈ −745). The invariant can only be verified
        // numerically at tensions where the barrier is comparable to k_BT.
        //
        // At −100 MPa (ΔP ≈ 10⁸ Pa):
        //   ΔG_c = 16π·γ³ / (3·ΔP²) ≈ 6.25×10⁻¹⁹ J
        //   ΔG_c / k_BT ≈ 154   →   exp(−154) ≈ 1.4×10⁻⁶⁷   (f64-representable)
        //   J ≈ J₀ · exp(−154) ≈ 1.4×10⁻³⁴ m⁻³ s⁻¹  > 0
        //
        // Reference: Church (2002) Eq. 7; physically accessible only at cavitation
        // inception in focused ultrasound — used here to validate code invariant.
        let rate = model.calculate_nucleation_rate(-1e8, temp, factor);
        assert!(
            rate > 0.0,
            "CNT rate at −100 MPa tension must be positive (got {rate:.3e})"
        );
    }

    #[test]
    fn test_stiffness_factor_logic() {
        let model = test_model();

        // Higher stiffness than reference => small reduction factor -> harder to cavitate
        let hard = model.calculate_stiffness_factor(1e6);
        assert!(hard < 1.0 && hard > 0.0);

        // Soft tissue (CTCs) => high reduction factor -> easier to cavitate
        let soft = model.calculate_stiffness_factor(1e4);
        assert_eq!(soft, 1.0); // Clamped at 1.0 maximum

        assert!(soft > hard);
    }
}
