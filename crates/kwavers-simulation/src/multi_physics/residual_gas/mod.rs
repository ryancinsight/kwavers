//! Residual cavitation-gas field and its coupling to acoustic medium properties.
//!
//! After a therapy pulse, the focal volume retains a microbubble cloud. Between
//! pulses that gas dissolves (Epstein–Plesset / shelled kinetics), and while it
//! persists it changes the effective medium the next pulse sees — collapsing the
//! sound speed (Wood) and adding resonant attenuation (Commander–Prosperetti).
//! This is the genuine, first-principles mechanism of residual-bubble shielding.
//!
//! [`ResidualGasField`] tracks the per-voxel gas void fraction `β(x)` and a
//! representative residual-bubble radius, evolves them with any
//! [`DissolutionModel`](kwavers_physics::acoustics::bubble_dynamics::DissolutionModel),
//! and produces the modified sound-speed and attenuation fields the solver
//! applies to the next pulse.

use kwavers_physics::acoustics::bubble_dynamics::{
    commander_prosperetti_attenuation, integrate_dissolution, wood_sound_speed, DissolutionModel,
};
use leto::{Array3, ArrayView3};

/// Per-voxel residual cavitation-gas void-fraction field with dissolution
/// kinetics and acoustic-property coupling.
///
/// The residual cloud is modelled as monodisperse at an evolving representative
/// radius: each pulse deposits fresh bubbles (resetting the representative
/// radius to the nucleation radius), and between pulses the population shrinks
/// per the chosen dissolution model, so `β` decays as `(R(t)/R₀)³` (gas-bubble
/// number conserved). This lumped model is exact for a monodisperse cloud and a
/// good approximation for a narrow size distribution.
#[derive(Debug, Clone)]
pub struct ResidualGasField {
    void_fraction: Array3<f64>,
    representative_radius_m: f64,
    deposit_radius_m: f64,
}

impl ResidualGasField {
    /// Create an empty field (`β = 0`) on a grid of the given shape, with the
    /// equilibrium radius of freshly nucleated bubbles `deposit_radius_m`.
    #[must_use]
    pub fn new(shape: (usize, usize, usize), deposit_radius_m: f64) -> Self {
        Self {
            void_fraction: Array3::zeros(shape),
            representative_radius_m: deposit_radius_m.max(1e-12),
            deposit_radius_m: deposit_radius_m.max(1e-12),
        }
    }

    /// Current gas void-fraction field `β(x)`.
    #[must_use]
    pub fn void_fraction(&self) -> ArrayView3<'_, f64> {
        self.void_fraction.view()
    }

    /// Current representative residual-bubble radius [m].
    #[must_use]
    pub fn representative_radius(&self) -> f64 {
        self.representative_radius_m
    }

    /// Deposit freshly nucleated cavitation gas: add the per-voxel gas volume
    /// fraction `gas_fraction` to `β` and reset the representative radius to the
    /// nucleation radius (each pulse refreshes the cloud).
    pub fn deposit(&mut self, gas_fraction: ArrayView3<'_, f64>) {
        if gas_fraction.shape() == self.void_fraction.shape() {
            for (b, g) in self.void_fraction.iter_mut().zip(gas_fraction.iter()) {
                *b = (*b + (*g).max(0.0)).min(1.0 - 1e-9);
            }
        }
        self.representative_radius_m = self.deposit_radius_m;
    }

    /// Evolve the residual cloud over a rest interval `dt_s` using `model`: the
    /// representative bubble shrinks per the dissolution kinetics and `β` is
    /// scaled by the cubed radius ratio (bubble number conserved). When the
    /// representative radius dissolves to zero the field is cleared.
    pub fn dissolve(&mut self, dt_s: f64, model: &impl DissolutionModel) {
        if !(dt_s.is_finite() && dt_s > 0.0) || self.representative_radius_m <= 0.0 {
            return;
        }
        let r0 = self.representative_radius_m;
        // Sub-step the dissolution ODE across the rest interval.
        let sub = (dt_s / 50.0).max(1e-9);
        let traj = integrate_dissolution(model, r0, sub, dt_s, 1e-12);
        let r_new = traj.radius.last().copied().unwrap_or(0.0).max(0.0);
        let factor = if r0 > 0.0 {
            (r_new / r0).powi(3).clamp(0.0, 1.0)
        } else {
            0.0
        };
        for b in self.void_fraction.iter_mut() {
            *b *= factor;
        }
        self.representative_radius_m = r_new;
    }

    /// Effective sound-speed field (Wood 1930) for the next pulse, given the
    /// liquid and gas acoustic properties.
    #[must_use]
    pub fn sound_speed_field(
        &self,
        c_liquid: f64,
        rho_liquid: f64,
        c_gas: f64,
        rho_gas: f64,
    ) -> Array3<f64> {
        self.void_fraction
            .mapv(|b| wood_sound_speed(b, c_liquid, rho_liquid, c_gas, rho_gas))
    }

    /// Excess attenuation field [Np/m] (Commander–Prosperetti) at the drive
    /// frequency, from the residual cloud at the representative radius.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn attenuation_field(
        &self,
        freq_hz: f64,
        c_liquid: f64,
        rho_liquid: f64,
        mu_liquid: f64,
        p0_pa: f64,
        polytropic: f64,
    ) -> Array3<f64> {
        let r0 = self.representative_radius_m;
        self.void_fraction.mapv(|b| {
            commander_prosperetti_attenuation(
                freq_hz, b, r0, c_liquid, rho_liquid, mu_liquid, p0_pa, polytropic,
            )
        })
    }

    /// Total residual gas volume `Σ β · dV` [m³] given the voxel volume.
    #[must_use]
    pub fn total_gas_volume(&self, dv_m3: f64) -> f64 {
        self.void_fraction.iter().sum::<f64>() * dv_m3.max(0.0)
    }

    /// Peak void fraction anywhere in the field.
    #[must_use]
    pub fn peak_void_fraction(&self) -> f64 {
        self.void_fraction.iter().copied().fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests;
