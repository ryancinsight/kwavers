// source/mod.rs

pub mod basic;
pub mod custom;
pub mod optical;
pub mod transducers;
pub mod wavefront;

pub mod config;
pub mod factory;
pub mod grid_source;
pub mod structs;
pub mod types; // New factory module

// Export traits and types
pub use config::{EnvelopeType, PulseParameters, PulseType, SourceModel, SourceParameters};
pub use factory::SourceFactory;
pub use grid_source::{GridSource, SourceMode};
pub use structs::{CompositeSource, NullSource, PointSource, TimeVaryingSource};
pub use types::{Source, SourceField, SourceType};

// Re-export submodules
pub use basic::{
    linear_array::LinearArray,
    matrix_array::MatrixArray,
    piston::{PistonApodization, PistonBuilder, PistonConfig, PistonSource},
};
pub use custom::{
    CustomSourceBuilder, FunctionSource, SimpleCustomSource, SimpleCustomSourceBuilder,
};
pub use optical::laser::{GaussianLaser, LaserConfig, LaserSource};
pub use transducers::{
    apodization::{
        Apodization, BlackmanApodization, GaussianApodization, HammingApodization,
        HanningApodization, RectangularApodization,
    },
    focused::{
        make_annular_array, make_bowl, ApodizationType, ArcConfig, ArcSource, BowlConfig,
        BowlTransducer, MultiBowlArray,
    },
    phased_array::{
        BeamformingMode, ElementSensitivity, PhasedArrayConfig, PhasedArrayTransducer,
        TransducerElement,
    },
    rectangular::RectangularTransducer,
};
pub use wavefront::{
    bessel::{BesselBuilder, BesselConfig, BesselSource},
    gaussian::{GaussianBuilder, GaussianConfig, GaussianSource},
    plane_wave::{PlaneWaveBuilder, PlaneWaveConfig, PlaneWaveSource},
    spherical::{SphericalBuilder, SphericalConfig, SphericalSource, SphericalWaveType},
};
