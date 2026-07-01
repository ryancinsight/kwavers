//! Signal-integrity vertical slice — controlled-impedance microstrip / stripline /
//! propagation / crosstalk kernels (Phase 3f carve).
//!
//! All eight free fns (4 characteristic-impedance + 1 propagation-delay + 1 skew-budget +
//! 1 risetime-degradation + 1 crosstalk coupling) take `f64` inputs and return `f64` outputs
//! with no internal state or cross-slice dependency on the rest of the physics tree. The
//! carved-per-concern submodules are:
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
//! # Phase 1a migration roadmap
//!
//! [`impedance_target`] + [`return_loss_db`] + [`microstrip_impedance`] etc. return `f64`
//! today. Phase 2 will replace the `w, h, er, t, b, s` parameters with `Meter, Meter, f64,
//! Meter, Meter, Meter` for the dimensioned quantities and return types as the typed
//! [`Ohm`] wrapper. **No signature change at Phase 3f** — keeping the API as `f64` preserves
//! every existing call-site and test fixture until the vertical-slice units land.
//!
//! [`Ohm`]: crate::units::Ohm
//!
//! # SSOT for the slice
//!
//! * `pub mod impedance` — characteristic impedance (microstrip + stripline + differential)
//!   + signal-line branching-match target + per-call RL for frequency-band loops.
//! * `pub mod propagation` — microstrip delay + cross-trace skew budget + risetime spread.
//! * `pub mod crosstalk` — backward-coupling coefficient + IEEE amplitude-ratio COM margin.
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
//! # Phase 3f cut-over status
//!
//! The flat `src/si.rs` (Phase 0 surface) has been retired: all 8 prior `pub fn`s have
//! migrated into the carved tree at this path, plus 3 new APIs (impedance_target,
//! return_loss_db, channel_operating_margin_db) added to fill out the frequency-band-aware
//! impedance budget surface. The crate-root re-export at `src/lib.rs::pub use physics::si::{...}`
//! covers the 10-fn surface for downstream callers; `crate::si::*` is now retired.

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
