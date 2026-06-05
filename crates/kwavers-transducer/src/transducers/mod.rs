//! Transducer source definitions
//!
//! This module contains transducer source definitions.

pub mod acquisition_geometry;
pub mod focused;
pub mod phased_array;
pub mod physics;
pub mod rectangular;

pub use acquisition_geometry::{ElementPosition, TransducerGeometry};
// Apodization windows now live in `kwavers_source::apodization`.
pub use focused::{
    make_annular_array, make_bowl, ApodizationType, ArcConfig, ArcSource, BowlConfig,
    BowlTransducer, MultiBowlArray, SphericalCapConfig, SphericalCapElement, SphericalCapLayout,
};
pub use phased_array::{
    BeamformingMode, ElementSensitivity, PhasedArrayConfig, PhasedArrayTransducer,
    TransducerElement,
};
pub use rectangular::RectangularTransducer;
