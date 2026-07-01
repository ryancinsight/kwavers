//! Placement energy: the DFM best practices as a single differentiable-enough cost the annealer
//! minimises. Each term is in millimetre units (areas in mm²) so the weights are dimensionless and
//! comparable; the annealer works on the weighted total.
//!
//! # Module layout
//!
//! * `config` — domain types: [`Axis`], [`PlaceWeights`], [`CongestionField`], [`PlaceConfig`], [`EnergyTerms`].
//! * `geom` — private geometry helpers used by the energy computation.
//! * `compute` — the [`energy`] orchestrator that calls the four term-accumulator modules.
//! * `thermal` — thermal-spread, IC-spread, and airflow-blockage terms.
//! * `proximity` — connector-EMI clearance, decoupling, termination, surge-suppressor, and crystal
//!   proximity terms.
//! * `connectivity` — HPWL, net-centre flight lines, signal-flow, crossing, and channel-blockage
//!   terms.
//! * `floorplan` — functional-region cohesion, board utilization, assembly alignment, LV↔HV
//!   isolation-barrier drift, and congestion feed-back terms.

pub mod compute;
pub mod config;
pub(super) mod connectivity;
pub(super) mod floorplan;
pub(crate) mod geom;
pub(super) mod proximity;
pub(super) mod thermal;

pub use compute::energy;
pub use config::{Axis, CongestionField, EnergyTerms, PlaceConfig, PlaceWeights};
