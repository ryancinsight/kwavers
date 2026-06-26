//! Placement energy: the DFM best practices as a single differentiable-enough cost the annealer
//! minimises. Each term is in millimetre units (areas in mm²) so the weights are dimensionless and
//! comparable; the annealer works on the weighted total.
//!
//! # Module layout
//!
//! * `config` — domain types: [`Axis`], [`PlaceWeights`], [`CongestionField`], [`PlaceConfig`], [`EnergyTerms`].
//! * `geom` — private geometry helpers used by the energy computation.
//! * `compute` — the [`energy`] function that evaluates all penalty terms.

pub mod config;
pub(crate) mod geom;
pub mod compute;

pub use config::{Axis, CongestionField, EnergyTerms, PlaceConfig, PlaceWeights};
pub use compute::energy;
