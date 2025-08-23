//! Transducer design and modeling
//!
//! Comprehensive transducer modeling following SOLID and SOC principles

pub mod backing;
pub mod coupling;
pub mod design;
pub mod directivity;
pub mod frequency;
pub mod geometry;
pub mod lens;
pub mod matching;
pub mod material;
pub mod sensitivity;

// Re-export core types
pub use backing::BackingLayer;
pub use coupling::ElementCoupling;
pub use design::TransducerDesign;
pub use directivity::DirectivityPattern;
pub use frequency::FrequencyResponse;
pub use geometry::ElementGeometry;
pub use lens::AcousticLens;
pub use matching::MatchingLayer;
pub use material::PiezoMaterial;
pub use sensitivity::TransducerSensitivity;