//! Per-domain physics models for driver-board design.
//!
//! Each sub-module owns one physical domain end to end — its types, its kernels, and its
//! tests — and the domains do not couple to one another: a slice depends only on
//! [`crate::geom`], [`crate::board`], and the unit newtypes in [`crate::units`], never on a
//! sibling slice. That isolation is what lets the placer and router consult any subset of
//! the physics without dragging the rest in.
//!
//! | Slice | Owns |
//! |---|---|
//! | [`ampacity`] | IPC-2221 conductor width and resistance, skin depth, current density, Black electromigration, plated-through-hole aspect ratio |
//! | [`dielectric`] | Paschen air breakdown, IPC-2221B external-conductor voltage spacing, relative CAF time-to-failure |
//! | [`thermal`] | 2-D heat conduction, IR drop, Joule electro-thermal coupling, thermal vias, transient time constant |
//! | [`emi`] | Commutation-loop and trace partial inductance, capacitive drive current, L·dI/dt overshoot, switching/gate/recovery loss, radiated estimate |
//! | [`pdn`] | Power-delivery IR-drop network, target impedance, hold-up capacitance, decoupling self-resonance |
//! | [`si`] | Microstrip impedance, propagation delay, skew, crosstalk, return loss, channel operating margin |
//! | [`acoustic`] | Wavelength, grating-lobe steering, focus and f-number, element directivity, BVD resonance, tissue attenuation, ISPPA |
//!
//! IR drop lives in [`thermal`] rather than [`pdn`] so the electro-thermal chain
//! (`ir_drop` → `joule_source` → `solve_electrothermal`) stays in one plane; [`pdn`] owns
//! the power-delivery impedance budget.
//!
//! Every public item here is re-exported at the crate root, so `crate::loop_inductance_nh`
//! and `crate::physics::emi::loop_inductance_nh` name the same function.

pub mod acoustic;
pub mod ampacity;
pub mod dielectric;
pub mod emi;
pub mod pdn;
pub mod si;
pub mod thermal;
