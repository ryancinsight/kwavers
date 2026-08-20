#![doc = include_str!("../README.md")]

pub mod array_2d;
pub mod basic;
pub mod beamforming;
pub mod bulk_piezo;
pub mod curvilinear;
pub mod design;
pub mod factory;
pub mod flexible;
pub mod hemispherical;
pub mod kwave_array;
pub mod mems;
mod parallel;
pub mod passive_acoustic_mapping;
pub mod transducers;
pub mod ultrafast;

// Aperture-driven design synthesis (size + frequency → element layout + wiring)
pub use design::{
    design_array, propagate_focused_linear_array, ApertureDesignSpec, ArrayDesign, ChannelWiring,
    FocusedLinearArrayPropagationSpec, FocusedPressureMap, DEFAULT_KERF_FRACTION,
    NYQUIST_PITCH_FRACTION,
};

// Source factory
pub use factory::SourceFactory;

// 2-D transducer arrays
pub use array_2d::{
    ApodizationType as Array2DApodizationType, ArrayCurvature, TransducerArray2D,
    TransducerArray2DBuilder, TransducerArray2DConfig,
};

// Basic array/element devices
pub use basic::{
    linear_array::LinearArray,
    matrix_array::MatrixArray,
    piston::{PistonApodization, PistonBuilder, PistonConfig, PistonSource},
};

// Flexible (calibrated) arrays
pub use flexible::{
    CalibrationData, CalibrationManager, FlexibleTransducerArray, FlexibleTransducerConfig,
    GeometrySnapshot,
};

// Hemispherical arrays
pub use hemispherical::{
    ArrayValidator, ElementConfiguration, ElementState, FocalPoint, HemisphereGeometry,
    HemisphericalArray, SparseArrayOptimizer, SteeringController,
};

// k-Wave array compatibility
pub use kwave_array::{DiscSourceProfile, ElementShape, KWaveArray};

// Transducer device families
pub use transducers::{
    acquisition_geometry::{ElementPosition, TransducerGeometry},
    focused::{
        make_annular_array, make_bowl, ApodizationType, ArcConfig, ArcSource, BowlConfig,
        BowlTransducer, MultiBowlArray, SphericalCapConfig, SphericalCapElement,
        SphericalCapLayout,
    },
    phased_array::{
        BeamformingMode, ElementSensitivity, PhasedArrayConfig, PhasedArrayTransducer,
        TransducerElement,
    },
    physics::{
        rayleigh_pressure, PlanarAperture, PlanarApertureGeometry, PlanarApertureShape,
        RayleighIntegralSpec, RayleighLayer, RayleighPropagationPath,
    },
    rectangular::RectangularTransducer,
};

// Receiver-array acquisition stacks
pub use beamforming::BeamformingCoreConfig;
pub use passive_acoustic_mapping::{PamArrayElement, PamArrayGeometry, PamDirectivityPattern};
