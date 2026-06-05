//! Electromagnetic wave equations and physics implementations
//!
//! This module provides the core electromagnetic physics implementations,
//! including Maxwell's equations, material properties, field calculations,
//! and source definitions.

pub mod fields;
pub mod materials;
pub mod traits;
pub mod types;

pub use fields::EMFieldUtils;
pub use materials::{EMMaterialDistribution, EMMaterialUtils};
pub use traits::{
    ElectromagneticWaveEquation, PhotoacousticCoupling, PhysicsEMSource,
    PlasmonicEnhancementEquation,
};
pub use types::{
    EMDimension, ElectromagneticPolarization, ElectromagneticWaveType, NanoparticleGeometry,
};
