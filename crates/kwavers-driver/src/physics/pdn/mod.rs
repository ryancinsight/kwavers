//! Power-delivery-network (PDN) decoupling, impedance and resonance — Phase 3e slice leaf.
//!
//! The IR-drop solver ([`crate::physics::thermal::IrDrop`] + [`crate::physics::thermal::ir_drop()`])
//! moved to the thermal slice at Phase 3b. This subtree keeps the **decoupling / resonance /
//! target-impedance / plane-cavity** half of PDN, organised as:
//!
//! * [`target_impedance`] — [`target_impedance::target_impedance_ohm`] +
//!   [`target_impedance::holdup_capacitance_f`] +
//!   [`target_impedance::max_decoupling_distance_mm`] (target-impedance budget + cap sizing +
//!   placement-budget derivation).
//! * [`impedance`] — [`impedance::self_resonant_freq_hz`] +
//!   [`impedance::pdn_impedance_at_freq`] +
//!   [`impedance::anti_resonance_hz`] (parallel-bank impedance + antiparallel LC peak).
//! * [`cavity`] — [`cavity::plane_resonance_hz`] (power-plane `(m, n)` cavity mode).
//!
//! All seven free functions are pure-math — they take `f64` inputs and return `f64` outputs, with
//! no internal state or cross-slice dependency. Migration motivates splitting them by physical
//! role (target-impedance sizing vs. parallel-bank impedance vs. plane cavity), not by file-size
//! symmetry. [`pdn_impedance_at_freq`] is the most likely target for a `(Farad, Ohm, Henry)` typed
//! struct at Phase 2 alongside the rest of the units migration.
//!
//! # Phase 1a migration roadmap
//!
//! [`crate::physics::thermal::IrDrop::max_drop_v`] is flat `f64` today. Phase 2 will replace it
//! with [`crate::units::Volt`] alongside the rest of the PDN signature migration (`supply` point
//! already carries `Nm`, so only the impedance / voltage / current parameters remain). The
//! `(C_f, ESR_ohm, ESL_h)` tuple in [`impedance::pdn_impedance_at_freq`] is the most likely target
//! for a `(Farad, Ohm, Henry)` typed struct at Phase 2.
//!
//! The acoustic output of a 150 V pulser scales with the delivered rail voltage, so resistive
//! voltage drop on VPP/GND between the supply connector and each device sets the channel-to-channel
//! **amplitude uniformity** of the array. This estimates the worst-case IR drop along the routed
//! power nets as a resistor network: each track segment is a conductance `g = 1/R` (R from
//! [`crate::physics::ampacity::track_resistance()`]); the supply pad is the voltage reference and
//! device pads draw current. Node voltages solve the same Laplace system as the thermal field
//! — `∇·(σ∇V) = −J` — by Gauss–Seidel over the routed graph. The IR-drop path itself lives in
//! the thermal slice (see [`crate::physics::thermal::ir_drop()`]) because `ir_drop` and the
//! Joule-heating source ([`crate::physics::thermal::joule_source()`]) both consume
//! [`crate::physics::ampacity::track_resistance()`], so co-locating them keeps the electro-thermal
//! coupling chain in one crate plane.
//!
//! # Vertical-slice convention
//!
//! Symbol-level API surface (the seven free fns + their docstring targets) is identical to the
//! prior flat `crate::pdn::…` shape. Internal helpers stay private to the slice (`pub(super)`
//! where a sibling importer needs access; never `pub(crate)` or `pub`). Downstream `lib.rs`
//! carries the canonical `pub use physics::pdn::{…}` re-export so the crate-root API does not
//! change at the call-site level.

pub mod cavity;
pub mod impedance;
pub mod target_impedance;

pub use cavity::plane_resonance_hz;
pub use impedance::{anti_resonance_hz, pdn_impedance_at_freq, self_resonant_freq_hz};
pub use target_impedance::{
    holdup_capacitance_f, max_decoupling_distance_mm, target_impedance_ohm,
};

#[cfg(test)]
mod tests;
