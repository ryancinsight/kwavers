//! Core medium trait combining all property interfaces
//!
//! This module defines the fundamental trait that all medium implementations
//! must satisfy to be used in simulations.

use super::{
    acoustic::AcousticProperties,
    bubble::{BubbleProperties, BubbleState},
    core::{ArrayAccess, CoreMedium},
    elastic::{ElasticArrayAccess, ElasticProperties},
    optical::OpticalProperties,
    thermal::{ThermalField, ThermalProperties},
    viscous::ViscousProperties,
};
use std::fmt::Debug;

/// Core trait combining all medium properties required for simulations
///
/// This trait represents a complete medium specification including acoustic,
/// elastic, thermal, optical, and viscous properties. All concrete medium
/// implementations must satisfy this trait.
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
    + OpticalProperties
    + ViscousProperties
    + Debug
    + Send
    + Sync
{
}

/// Blanket implementation for any type satisfying all requirements
impl<T> Medium for T
where
    T: CoreMedium
        + ArrayAccess
        + AcousticProperties
        + BubbleProperties
        + BubbleState
        + ElasticProperties
        + ElasticArrayAccess
        + ThermalProperties
        + ThermalField
        + OpticalProperties
        + ViscousProperties
        + Debug
        + Send
        + Sync,
{
}