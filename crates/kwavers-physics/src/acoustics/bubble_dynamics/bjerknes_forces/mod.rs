//! Bjerknes Forces: Bubble-Bubble Interactions in Acoustic Fields
//!
//! This module implements primary and secondary Bjerknes forces that govern
//! the interactions between microbubbles in acoustic fields.
//!
//! ## Physical Background
//!
//! **Primary Bjerknes Force**:
//! Arises from the time-averaged pressure gradient in the acoustic field acting on
//! the oscillating bubble. As implemented in [`primary`]:
//!
//! ```text
//! F_primary = -(4π/3) R³ · ∇p
//! ```
//! the bubble volume times the (negative) local pressure gradient
//! (Blake 1986; Leighton 1994 §3.4).
//!
//! **Secondary Bjerknes Force**:
//! Results from the acoustic field scattered by one bubble affecting another.
//! This force can be attractive (bubbles oscillate in phase) or repulsive
//! (out of phase oscillations). As implemented in [`secondary`], the
//! time-averaged harmonic form derived from
//! `F = -(ρ₀/(4πd²)) ⟨V̇₁ V̇₂⟩`:
//!
//! ```text
//! F_secondary = -(ρ₀ ω² / (8π)) · ΔV₁ · ΔV₂ · cos(φ) / d²
//! ```
//! where ΔV₁, ΔV₂ are the bubbles' volume-oscillation amplitudes, φ is
//! the phase difference and d the separation. The scalar is the
//! line-of-centres component of the force on one bubble due to the
//! other, positive away from the other bubble — so in-phase pairs
//! (cos φ > 0) give a negative (attractive) force. The instantaneous
//! Crum 1975 form `F_B(t) = -ρ₀ V̇₁ V̇₂ / (4πd²)` is the canonical
//! kernel [`secondary_bjerknes_force_instantaneous`] in the same
//! submodule; `crate::acoustics::bubble_dynamics::interactions`
//! delegates to it (KW-PHYS-009 item 1, consolidated 2026-11-27 —
//! the two implementations previously carried opposite signs).
//!
//! ## Validity regime
//!
//! The time-averaged forms assume linear (small-amplitude)
//! oscillations and weak coupling. The [`regime`] submodule provides
//! the guards: the linear-regime displacement check (ξ/R₀ < 1) and
//! the measured aggregation window of Duryea et al. (2015)
//! (0.2 < MI < 1; aggregation fails above MI ≈ 1). Use
//! [`secondary_bjerknes_force_checked`] where the regime matters.
//!
//! ## References
//!
//! - Bjerknes, V. (1906). "Fields of force", Columbia University Press
//! - Crum, L. A. (1975). "Bjerknes forces on bubbles in a stationary sound field"
//! - Leighton, T. G. (1994). "The Acoustic Bubble", §3.4
//! - Duryea, A. P. et al. (2015). IEEE Trans. UFFC 62:1605–1614
//! - Pozarowski, P. & Holyst, R. (2002) "Two and three body interactions"
//! - Garbin, V. et al. (2007) "Dynamics of coated bubbles monitored by individual pulse acoustography"

pub mod calculator;
pub mod kinematics;
pub mod primary;
pub mod regime;
pub mod secondary;
pub mod types;

#[cfg(test)]
mod tests;

pub use calculator::BjerknesCalculator;
pub use primary::primary_bjerknes_force;
pub use regime::{
    bjerknes_regime, displacement_amplitude, duryea_aggregation_regime,
    radiated_pressure_amplitude, secondary_bjerknes_force_checked, AggregationRegime,
    BjerknesRegime, BJERKNES_LINEAR_MAX_DISPLACEMENT_RATIO, DURYEA_AGGREGATION_MI_MAX,
    DURYEA_AGGREGATION_MI_MIN,
};
pub use secondary::{secondary_bjerknes_force_instantaneous, secondary_bjerknes_force_mean};
pub use types::{BjerknesConfig, BjerknesForceData, BjerknesInteractionType};
