//! Theranostic orchestration: interleaved therapy + diagnostic monitoring.
//!
//! This module hosts the *orchestration* algorithms that tie the existing
//! kwavers physics and inverse-problem engines into a closed therapy + imaging
//! loop, as used by the `brain_theranostic_monitor` example:
//!
//! ```text
//!   CT  →  HU→c,ρ  →  1024-element hemisphere  →  FWI / RTM / Born
//!                                                       │ full-brain image
//!                                                       ▼
//!                                                  target selection  (targeting)
//!                                                       │
//!                          ┌────────────────────────────┴───────────────┐
//!                          ▼                                            ▼
//!            interleaved pulse schedule (pulsing)          lesion → medium coupling (lesion)
//!            · therapy bursts focused at target            · thermal: Δc(T) (Duck 1990)
//!            · sparse-subset imaging pulses (low dose)     · cavitation: Wood c(β) (Wood 1930)
//!                          │                                            │
//!                          └──────────────► monitored-slice recon ◄─────┘
//!                                            (RTM reflectivity + differential FWI)
//! ```
//!
//! The submodules here are **pure, deterministic, unit-tested** algorithms with
//! no I/O and no solver state — the heavy forward/adjoint simulations are run by
//! the existing `kwavers_solver` engines and fed through these helpers. Keeping
//! the orchestration logic separate from the example `main` makes it testable
//! against analytical references (the design constraint that the simulation must
//! be real and not faked to produce a desired image).
//!
//! # Submodules
//!
//! - [`pulsing`]  — low-dose sparse-transmit subsets + therapy/imaging interleave.
//! - [`lesion`]   — lesion → acoustic-medium perturbation (thermal and cavitation),
//!   the physical change the monitor reconstructs as the lesion grows.
//! - [`targeting`] — pick a sonication target from a reconstructed property volume.

pub mod lesion;
pub mod monitor;
pub mod pulsing;
pub mod targeting;

pub use lesion::{
    cavitation_perturbed_sound_speed, lesion_mask, perturb_sound_speed,
    thermal_perturbed_sound_speed, LesionState, TherapyMode, ABLATION_CEM43_THRESHOLD_MIN,
};
pub use pulsing::{interleave_schedule, sparse_transmit_subsets, PulseFrame, PulseKind};
pub use targeting::{voxel_to_position, TargetSelection};
