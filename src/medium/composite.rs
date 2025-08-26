//! Composite medium trait for full-featured media
//!
//! This module provides a composite trait that combines all medium properties,
//! serving as a migration path from the monolithic Medium trait.

use crate::medium::{
    acoustic::AcousticProperties,
    bubble::{BubbleProperties, BubbleState},
    core::{ArrayAccess, CoreMedium},
    elastic::{ElasticArrayAccess, ElasticProperties},
    optical::OpticalProperties,
    thermal::{TemperatureState, ThermalProperties},
    viscous::ViscousProperties,
};
use std::fmt::Debug;

/// Composite trait combining all medium properties
///
/// This trait is provided for backward compatibility and convenience when
/// a medium needs to support all physical properties. New code should prefer
/// using specific trait bounds for better modularity.
pub trait CompositeMedium:
    CoreMedium
    + ArrayAccess
    + AcousticProperties
    + ElasticProperties
    + ElasticArrayAccess
    + ThermalProperties
    + TemperatureState
    + OpticalProperties
    + ViscousProperties
    + BubbleProperties
    + BubbleState
    + Debug
    + Sync
    + Send
{
}

/// Blanket implementation for any type that implements all the required traits
impl<T> CompositeMedium for T where
    T: CoreMedium
        + ArrayAccess
        + AcousticProperties
        + ElasticProperties
        + ElasticArrayAccess
        + ThermalProperties
        + TemperatureState
        + OpticalProperties
        + ViscousProperties
        + BubbleProperties
        + BubbleState
        + Debug
        + Sync
        + Send
{
}

/// Legacy Medium trait for backward compatibility
///
/// This trait is now just an alias for CompositeMedium. All methods are
/// provided through the component traits. This avoids method ambiguity
/// while maintaining backward compatibility.
pub trait Medium: CompositeMedium {}

/// Blanket implementation for backward compatibility
impl<T: CompositeMedium> Medium for T {}
