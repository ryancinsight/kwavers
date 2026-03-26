//! Plasmonic Enhancement Implementations
//!
//! This module implements robust plasmonic enhancement models detailing
//! electromagnetic field scattering and absorption near metallic nanostructures.
//! Applications primarily concern photoacoustic imaging and therapy.
//!
//! Sub-modules encapsulate single-particle Mie theory, collective ensemble
//! effective media, and coherent nano-array lattice computations.

pub mod enhancement;
pub mod mie_theory;
pub mod nanoparticle_array;
pub mod types;

#[cfg(test)]
mod tests;

pub use enhancement::PlasmonicEnhancement;
pub use mie_theory::MieTheory;
pub use nanoparticle_array::NanoparticleArray;
pub use types::{ArrayGeometry, CouplingModel};
