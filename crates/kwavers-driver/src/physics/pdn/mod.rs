//! Power-delivery-network (PDN) decoupling, impedance and resonance.
//!
//! This subtree owns the **decoupling / resonance / target-impedance / plane-cavity** half
//! of PDN; the IR-drop solver ([`crate::physics::thermal::IrDrop`] +
//! [`crate::physics::thermal::ir_drop()`]) lives in the thermal slice for the reason given
//! below. The organisation is:
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
//! All seven free functions are pure math — `f64` in, `f64` out, no internal state and no
//! cross-slice dependency. They are grouped by physical role (target-impedance sizing vs.
//! parallel-bank impedance vs. plane cavity), not by file size.
//!
//! # Units
//!
//! Impedance, voltage, and current parameters are plain `f64` in their documented SI units;
//! only the `supply` point carries an [`crate::units::Nm`]. Typing them — the
//! `(C_f, ESR_ohm, ESL_h)` tuple in [`impedance::pdn_impedance_at_freq`] as a
//! `(Farad, Ohm, Henry)` struct, and [`crate::physics::thermal::IrDrop::max_drop_v`] as a
//! [`crate::units::Volt`] — is tracked in `docs/MIGRATION.md`.
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
//! # Visibility
//!
//! Internal helpers stay private to the slice — `pub(super)` where a sibling needs access,
//! never `pub(crate)` or `pub`. `lib.rs` carries the canonical `pub use physics::pdn::{…}`
//! re-export, so every public item here is also reachable from the crate root.

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
