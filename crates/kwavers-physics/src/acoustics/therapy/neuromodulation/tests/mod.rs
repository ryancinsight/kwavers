//! Value-semantic test suite for the neuromodulation module, organised by
//! concern: membrane models, capacitance sources, NICE/SONIC coupling, and
//! pulse-train protocol/dosimetry. Common imports are re-exported here so each
//! concern file can `use super::*`.

mod coupling;
mod membrane;
mod protocol;
mod sources;

pub(super) use super::hodgkin_huxley::{
    alpha_m, alpha_n, beta_h, simulate_hh, HhParams, HhState, HhTrace,
};
pub(super) use super::intramembrane_cavitation::{
    modulation_depth_from_pressure, CapacitanceModulation, BILAYER_AREA_MODULUS_N_M,
};
pub(super) use super::nice::{simulate_nice, NiceConfig};

/// Shared squid-HH resting potential used across the membrane and coupling
/// test groups [mV].
pub(super) const V_REST: f64 = -65.0;
