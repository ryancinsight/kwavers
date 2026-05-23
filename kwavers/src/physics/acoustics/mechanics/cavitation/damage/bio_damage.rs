//! # Biological Damage Models for Cavitation
//!
//! [PHASE-2: SPEC-DRIVEN IMPLEMENTATION]
//!
//! This module provides the mathematical formulations for cellular injury
//! due to hydrodynamic and acoustic cavitation. It models specific injury modes
//! such as cellular lysis, necrosis, and permeabilization using cumulative dose
//! metrics and peak event thresholds.
//!
//! ## Mathematical Specifications
//!
//! ### Theorem 1: Cellular Lysis Probability
//! The probability of immediate cellular lysis $P_L$ due to discrete cavitation collapse
//! events is bounded by the local pressure gradient across the cell membrane:
//! $$ P_L = 1 - \exp\left( - \frac{W_{collapse} - W_{threshold}}{\tau_{lysis}} \right) $$
//! for $W_{collapse} > W_{threshold}$, where $W$ is the collapse energy density.
//!
//! ### Theorem 2: Necrosis Accumulation
//! Fractional tissue necrosis fraction $F_N$ accumulates via the cavitation dose $D_c$:
//! $$ \frac{d F_N}{dt} = k_n (1 - F_N) D_c $$
//!
//! ### Invariants
//! 1. Lysis probability satisfies $0.0 \le P_L \le 1.0$.
//! 2. Necrosis fraction monotonically increases: $\Delta F_N \ge 0.0$.
//! 3. No damage occurs below mechanical thresholds: $W_{collapse} < W_{threshold} \implies P_L = 0.0$.
//!
//! ## Literature
//! - Duck, F. A. (1990). Physical Properties of Tissue. Academic Press.
//! - Apfel, R. E., & Holland, C. K. (1991). Gauging the likelihood of cavitation from short-pulse, low-duty cycle diagnostic ultrasound.

use crate::domain::medium::properties::AcousticMaterialProperties;

/// Trait defining the contract for calculating cavitation-induced biological damage.
pub trait BioDamageModel {
    /// Required floating point precision for the implementation.
    type Scalar;

    /// Calculates the instantaneous lysis probability from a cavitation event.
    ///
    /// # Parameters
    /// - `collapse_energy`: The volumetric energy density released during bubble collapse ($J/m^3$).
    /// - `material_properties`: The properties of the host tissue determining the threshold.
    ///
    /// # Returns
    /// Probability $P_L \in [0.0, 1.0]$.
    fn calculate_lysis_probability(
        &self,
        collapse_energy: Self::Scalar,
        material_properties: &AcousticMaterialProperties,
    ) -> Self::Scalar;

    /// Tracks the accumulation of necrosis over a time interval $\Delta t$.
    ///
    /// # Parameters
    /// - `current_necrosis`: The existing fraction of necrotic tissue $F_N \in [0.0, 1.0]$.
    /// - `cavitation_dose`: The cumulative dose metric $D_c$ observed over the interval.
    /// - `dt`: The length of the time integration step.
    ///
    /// # Returns
    /// The updated necrosis fraction $F_N$.
    fn update_necrosis_fraction(
        &self,
        current_necrosis: Self::Scalar,
        cavitation_dose: Self::Scalar,
        dt: Self::Scalar,
    ) -> Self::Scalar;
}

/// Empirical biological damage model following Duck 1990 acoustic limits.
///
/// Implements immediate lysis thresholds and time-integrated necrosis calculations.
#[derive(Debug, Clone)]
pub struct EmpiricalBioDamageModel<T: std::fmt::Debug + Clone> {
    /// Lysis threshold energy $W_{threshold}$ ($J/m^3$).
    pub lysis_threshold: T,
    /// Lysis time constant $\tau_{lysis}$ scale factor.
    pub lysis_tau: T,
    /// Necrosis accumulation rate $k_n$.
    pub necrosis_rate: T,
}

impl<T: num_traits::Float + std::fmt::Debug + Clone> BioDamageModel for EmpiricalBioDamageModel<T> {
    type Scalar = T;

    #[inline(always)]
    fn calculate_lysis_probability(
        &self,
        collapse_energy: Self::Scalar,
        _material: &AcousticMaterialProperties,
    ) -> Self::Scalar {
        if collapse_energy <= self.lysis_threshold {
            T::zero()
        } else {
            let one = T::one();
            let exponent = -(collapse_energy - self.lysis_threshold) / self.lysis_tau;
            one - exponent.exp()
        }
    }

    #[inline(always)]
    fn update_necrosis_fraction(
        &self,
        current_necrosis: Self::Scalar,
        cavitation_dose: Self::Scalar,
        dt: Self::Scalar,
    ) -> Self::Scalar {
        let one = T::one();
        let delta = self.necrosis_rate * (one - current_necrosis) * cavitation_dose * dt;
        let next_val = current_necrosis + delta;

        // Manual clamp to avoid missing primitive bounds depending on trait presence
        if next_val > one {
            one
        } else if next_val < T::zero() {
            T::zero()
        } else {
            next_val
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;

    fn test_model() -> EmpiricalBioDamageModel<f64> {
        EmpiricalBioDamageModel {
            lysis_threshold: 100.0,
            lysis_tau: 20.0,
            necrosis_rate: 0.05,
        }
    }

    #[test]
    fn test_lysis_probability_bounds() {
        let model = test_model();
        let props =
            AcousticMaterialProperties::new(SOUND_SPEED_WATER_SIM, 1000.0, 0.0, 4184.0, 0.6);

        // Below threshold is zero
        assert_eq!(model.calculate_lysis_probability(50.0, &props), 0.0);

        // Exactly threshold is zero
        assert_eq!(model.calculate_lysis_probability(100.0, &props), 0.0);

        // Above threshold increases monotonically asymptotically to 1.0
        let p1 = model.calculate_lysis_probability(110.0, &props);
        let p2 = model.calculate_lysis_probability(150.0, &props);

        assert!(p1 > 0.0 && p1 < 1.0);
        assert!(p2 > p1 && p2 < 1.0);

        // Very high energy gives practically 1.0
        let p_max = model.calculate_lysis_probability(1000.0, &props);
        assert!((p_max - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_necrosis_monotonic_accumulation() {
        let model = test_model();

        // Zero dose -> zero delta
        assert_eq!(model.update_necrosis_fraction(0.1, 0.0, 1.0), 0.1);

        // Active dose -> positive accumulation
        let next = model.update_necrosis_fraction(0.1, 10.0, 0.5);
        assert!(next > 0.1);

        // Cannot exceed 1.0
        let saturated = model.update_necrosis_fraction(0.99, 1000.0, 10.0);
        assert_eq!(saturated, 1.0);
    }
}
