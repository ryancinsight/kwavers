//! Low-level acoustic/optical/EM excitation primitives for kwavers.
//!
//! This crate owns the fundamental source abstractions and field profiles —
//! the [`Source`] trait, grid/mask-driven sources, point/composite/time-varying
//! sources, analytic wavefronts (plane/Bessel/Gaussian/spherical), custom
//! arbitrary-signal sources, electromagnetic/optical primitives, and the
//! apodization-window math shared by higher-level devices.
//!
//! Physical transducer *devices* (bowls, phased/linear/matrix arrays,
//! calibration, factories) live in `kwavers-transducer`, which depends on this
//! crate and on `kwavers-receiver`.

pub mod apodization;
pub mod config;
pub mod custom;
pub mod electromagnetic;
pub mod grid_source;
pub mod injection;
pub mod optical;
pub mod structs;
pub mod types;
pub mod wavefront;

// Core source trait and value types
pub use config::{
    DomainSourceParameters, EnvelopeType, FocusedBowlAperture, PulseParameters, PulseType,
    SourceModel,
};
pub use grid_source::{GridSource, SourceMode};
pub use injection::SourceInjectionMode;
pub use structs::{CompositeSource, NullSource, PointSource, TimeVaryingSource};
pub use types::{Source, SourceEMWaveType, SourceField, SourcePolarization, SourceType};

// Apodization windows (shared low-level math)
pub use apodization::{
    Apodization, BlackmanApodization, GaussianApodization, HammingApodization, HanningApodization,
    RectangularApodization, TukeyApodization,
};

// Custom / arbitrary-signal sources
pub use custom::{
    CustomSourceBuilder, FunctionSource, SimpleCustomSource, SimpleCustomSourceBuilder,
};

// Electromagnetic and optical primitives
pub use electromagnetic::{DomainEMSource, PlaneWaveEMSource, PointEMSource};
pub use optical::laser::{GaussianLaser, LaserConfig, LaserSource};

// Analytic wavefronts
pub use wavefront::{
    bessel::{BesselBuilder, BesselConfig, BesselSource},
    gaussian::{GaussianBuilder, GaussianConfig, GaussianSource},
    plane_wave::{InjectionMode, PlaneWaveBuilder, PlaneWaveSource, PlaneWaveSourceConfig},
    spherical::{SphericalBuilder, SphericalConfig, SphericalSource, SphericalWaveType},
};
