#![doc = include_str!("../README.md")]

pub mod apodization;
pub mod config;
pub mod custom;
pub mod electromagnetic;
pub mod grid_source;
pub mod injection;
pub mod optical;
mod parallel;
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
