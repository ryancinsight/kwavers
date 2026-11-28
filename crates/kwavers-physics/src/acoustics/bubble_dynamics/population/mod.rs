//! Bubble-population operators: fragmentation and coalescence.
//!
//! # Status: OWN CONSTRUCTION — no literature law
//!
//! Unlike the diffusion and force laws elsewhere in this crate,
//! there is **no published closed-form law** for how a cavitation
//! bubble population redistributes itself by fragmentation and
//! coalescence. The operators here are the formation model's own
//! construction (Builds 5–7 of the triplet-pulse formation work,
//! `formation/b5_fragmentation.py`, `formation/b7_*.py`), written to
//! conserve gas exactly and to expose their assumptions as
//! parameters. They are bookkeeping operators, not physics claims:
//!
//! - **Fragmentation** ([`fragmentation`]): a fraction of a group
//!   breaks into daughters distributed by a caller-chosen kernel
//!   (lognormal-in-radius or equal-split). Whether fragmentation
//!   occurs at all is gated by the parent's collapse ratio against
//!   the Chomas et al. (2001) indicator — that indicator is
//!   literature; the daughter distribution is not. **Single-
//!   sequence/single-burst use only** — per-sequence iteration
//!   over repeated exposures has no fixed point (Build 11).
//! - **Coalescence** ([`coalescence`]): pairs merge with a
//!   probability from the geometric neighbour estimate. The merge
//!   conserves gas and volume. **Application discipline matters:**
//!   re-applying the operator every cycle with re-randomised pairing
//!   is unphysical for bubbles fixed in tissue (Build 7 verdict) —
//!   the physical use is one-shot (or with tracked pairing). Above
//!   MI ≈ 1 coalescence during a burst is additionally excluded by
//!   the Duryea et al. (2015) cliff — see
//!   [`bjerknes_forces::regime`](crate::acoustics::bubble_dynamics::bjerknes_forces::regime).
//!
//! A per-cycle fragmentation probability of 0.02 appears in the
//! prototype (Stage 7) as [`fragmentation::P_FRAG_FITTED_LI_2021`]:
//! it is a **fitted** calibration to Li et al. (2021), not a
//! prediction, and is labelled as such.
//!
//! Radii in a group are tied to gas content through the equilibrium
//! (residual) map of
//! [`bader_influx`](crate::acoustics::mechanics::cavitation::core::bader_influx):
//! a bubble holding `n` molecules at ambient pressure sits at the
//! radius solving `R³(p₀ + 2σ/R) = 3k_BT·n/(4π)`.

pub mod coalescence;
pub mod fragmentation;
mod group;
pub mod imaging_evolution;
mod medium;

pub use coalescence::{coalesce_group, merge_pair, neighbour_merge_probability};
pub use fragmentation::{
    collapse_ratio, EqualSplitKernel, FragmentationKernel, LognormalKernel,
    CHOMAS_FRAGMENTATION_DIAMETER_RATIO, P_FRAG_FITTED_LI_2021,
};
pub use group::BubbleGroup;
pub use imaging_evolution::{
    epstein_plesset_step, evolve_under_imaging_train, EpsteinPlessetMedium, ImagingEvolution,
    ImagingPulseEffect, ImagingTrain, DISSOLUTION_REMOVAL_RADIUS_M,
    IMAGING_FRAGMENTATION_GATE_RMAX_R0,
};
pub use medium::PopulationMedium;
