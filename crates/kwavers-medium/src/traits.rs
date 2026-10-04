//! Core medium trait combining all property interfaces
//!
//! This module defines the fundamental trait that all medium implementations
//! must satisfy to be used in simulations.

use super::{
    acoustic::AcousticProperties,
    bubble::{BubbleProperties, BubbleState},
    core::{ArrayAccess, CoreMedium},
    elastic::{ElasticArrayAccess, ElasticProperties},
    optical::MediumOpticalProperties,
    thermal::{ThermalField, ThermalProperties},
    viscous::ViscousProperties,
};
use std::fmt::Debug;

/// Core trait combining all medium properties required for simulations
///
/// This trait represents a complete medium specification including acoustic,
/// elastic, thermal, optical, and viscous properties. All concrete medium
/// implementations must satisfy this trait.
// dyn: used as `&dyn Medium` (open implementor set: the config-driven
// MediumType family built by MediumBuilder plus domain-specific loaders).
// Per the zero-cost policy (ADR 012) this is a sanctioned dynamic-dispatch
// boundary: the hot paths extract medium properties at construction
// (initialize_field_arrays, the construction orchestrator) and the PSTD
// stepper holds no medium reference at all; the remaining runtime calls are
// per-thermal-update scalar fetches (thermal_diffusivity), never per-cell.
pub trait Medium:
    CoreMedium
    + ArrayAccess
    + AcousticProperties
    + BubbleProperties
    + BubbleState
    + ElasticProperties
    + ElasticArrayAccess
    + ThermalProperties
    + ThermalField
    + MediumOpticalProperties
    + ViscousProperties
    + Debug
    + Send
    + Sync
{
}

/// Blanket implementation for any type satisfying all requirements
impl<T> Medium for T where
    T: CoreMedium
        + ArrayAccess
        + AcousticProperties
        + BubbleProperties
        + BubbleState
        + ElasticProperties
        + ElasticArrayAccess
        + ThermalProperties
        + ThermalField
        + MediumOpticalProperties
        + ViscousProperties
        + Debug
        + Send
        + Sync
{
}
