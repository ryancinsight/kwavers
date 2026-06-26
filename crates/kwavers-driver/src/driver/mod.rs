//! Transducer-driver power-loss, efficiency, and matching-network physics.
//!
//! The optimisation needs to know *where the heat is* and *how efficient the drive is*, both of which
//! come from the pulser loss model — not a flat per-IC guess. This module derives, from first
//! principles, the per-channel dissipation of a class-D HV pulser driving a (largely capacitive)
//! piezoelectric load, splits it between the device and the series damping resistor (so each becomes a
//! correctly-weighted thermal source for placement), and quantifies the electrical→acoustic
//! efficiency and what an output matching network would buy.
//!
//! # Loss terms (per channel, duty-cycle weighted)
//! * **Dynamic / switching** `P_dyn = D·f·C_load·V²` — charge+discharge of the clamped capacitance
//!   each drive cycle. For an R-charged capacitor the dissipated energy is ½CV² per edge regardless
//!   of R, so this term is fixed by `C, V, f` and then **split** between the device on-resistance and
//!   the series resistor in proportion to their resistance (series current path).
//! * **Gate drive** `P_g = D·Q_g·V_drv·f` and **reverse recovery** `P_rr = D·Q_rr·V·f` — both
//!   `Q·V·f`, device-side.
//!
//! # Matching
//! A direct pulser dissipates essentially all of `C₀·V²·f` (no energy recovery), so its electrical
//! efficiency into a capacitive load is poor. A series/parallel **tuning inductor** resonating out
//! `C₀` at `f₀` recovers that reactive energy, leaving only the radiation-resistance loss — the
//! quantitative case for a matching network. The series resistor in the present design is instead a
//! **damping** element that sets the ringdown `Q` (short imaging pulses), a different objective.
//!
//! Evidence tier: closed-form circuit physics (CV²f, Q·V·f, LC resonance, RC ringdown), value-
//! semantic unit tests against hand-computed references.
//!
//! # Slice layout
//!
//! Carved by **physics role** (Phase 4g). Plain backticks name the slice-private submodules; the
//! public items each hosts stay clickable.
//! * `pulser` — the core loss model: [`PulserOp`] operating point → [`PulserDissipation`] breakdown
//!   via [`pulser_dissipation`].
//! * `reactive` — matching-network / reactive-drive / ringdown math
//!   ([`tuning_inductor_h`], [`load_quality_factor`], [`damping_resistor_ohm`], [`ringdown_cycles`],
//!   [`reactive_drive_power_w`], [`driver_efficiency`], [`switching_node_ringing_v`]).
//! * `rating` — thermal-duty + package-power-rating limits ([`max_safe_duty`], [`chip_power_rating_w`],
//!   [`power_rating_check`], [`thermally_derated_efficiency`]).
//! * `sweep` — the frequency-sweep driver-loss optimiser ([`FreqSweepPoint`], [`sweep_driver_loss`],
//!   [`find_best_freq`]).
//! * `compare` — cross-IC comparison at one operating point ([`ComponentComparison`],
//!   [`compare_driver_ics_at`]).

mod compare;
mod pulser;
mod rating;
mod reactive;
mod sweep;

#[cfg(test)]
mod tests;

pub use compare::{compare_driver_ics_at, ComponentComparison};
pub use pulser::{pulser_dissipation, PulserDissipation, PulserOp};
pub use rating::{
    chip_power_rating_w, max_safe_duty, power_rating_check, thermally_derated_efficiency,
    PowerOverload, PowerRatingReport,
};
pub use reactive::{
    damping_resistor_ohm, driver_efficiency, load_quality_factor, reactive_drive_power_w,
    ringdown_cycles, switching_node_ringing_v, tuning_inductor_h,
};
pub use sweep::{find_best_freq, sweep_driver_loss, FreqSweepPoint};

/// Nominal junction-to-case thermal resistance (K/W) for a typical HV-class pulser IC in a
/// SOIC-8 or QFN-class package, used to convert device dissipation to a temperature rise in
/// [`compare_driver_ics_at`] and [`sweep_driver_loss`]. Source: typical datasheet θ_jc values
/// for 8–10 mm² footprint power packages at 1 W steady dissipation. Slice-private (`pub(super)`)
/// so the `sweep` and `compare` sub-files share one source of truth without exposing it on the
/// `crate::driver` surface.
pub(super) const DEFAULT_THETA_JC_K_PER_W: f64 = 40.0;
