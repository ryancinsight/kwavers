//! Signal integrity: controlled-impedance microstrip and stripline, propagation, and
//! crosstalk kernels.
//!
//! Every function here is pure math — `f64` in, `f64` out, no internal state and no
//! cross-slice dependency on the rest of the physics tree. The per-concern submodules are:
//!
//! * [`impedance`] — [`impedance::microstrip_eeff`] + [`impedance::microstrip_impedance`] +
//!   [`impedance::stripline_impedance`] + [`impedance::differential_microstrip_impedance`] +
//!   [`impedance::impedance_target`] (signal-line branching-match target) +
//!   [`impedance::return_loss_db`] (single-freq RL for caller-loop iteration over freq bands).
//! * [`propagation`] — [`propagation::microstrip_delay_s_per_m`] +
//!   [`propagation::within_skew`] + [`propagation::risetime_degradation_ps_per_m`] (the
//!   timing-half of signal integrity: per-metre delay, length-matching skew budget, and
//!   skin/dielectric-driven edge spread).
//! * [`crosstalk`] — [`crosstalk::crosstalk_coupling`] +
//!   [`crosstalk::channel_operating_margin_db`] (IEEE amplitude-ratio COM for the
//!   coupled-line noise floor — this is the neighbour of [`crosstalk_coupling`] because the
//!   two compose into the eye-mask budget check that the polyline router exercises vs the
//!   receiver threshold).
//!
//! # Units
//!
//! The dimensioned parameters (`w`, `h`, `t`, `b`, `s`) and the impedance return values are
//! plain `f64` in their documented SI units. Typing them as lengths and [`Ohm`] is tracked
//! in `docs/MIGRATION.md`.
//!
//! [`Ohm`]: crate::units::Ohm
//!
//! # SSOT distinction with PDN
//!
//! The signal-line [`impedance::impedance_target`] (driver Z + tolerated Γ) is **distinct**
//! from the PDN power-rail [`crate::physics::pdn::target_impedance_ohm`] (V_tolerance / I_step).
//! The two functions solve different physical problems at different impedance scales (SI
//! operates at 25–100 Ω of controlled-impedance routing; PDN operates at single-digit mΩ of
//! bulk decoupling) and must not be substituted for each other at a call site. The
//! distinguished SSOT is anchored in the `crate::physics::si::tests::ssot_distinction_pdn_target_impedance_is_separate`
//! test fixture in the slice's consolidated `tests.rs`.
//!
//! Every public item here is re-exported at the crate root by
//! `src/lib.rs::pub use physics::si::{…}`.

pub mod crosstalk;
pub mod impedance;
pub mod propagation;

pub use crosstalk::{channel_operating_margin_db, crosstalk_coupling};
pub use impedance::{
    differential_microstrip_impedance, impedance_target, microstrip_impedance, return_loss_db,
    stripline_impedance,
};
pub use propagation::{microstrip_delay_s_per_m, risetime_degradation_ps_per_m, within_skew};

#[cfg(test)]
mod tests;
