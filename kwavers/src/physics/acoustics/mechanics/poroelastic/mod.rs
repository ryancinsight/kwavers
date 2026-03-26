//! Poroelastic Tissue Modeling - Biphasic Fluid-Solid Coupling
//!
//! Implements Biot theory for wave propagation in porous media with
//! applications to biological tissue modeling.
//!
//! ## Overview
//!
//! Poroelastic materials consist of:
//! 1. **Solid Matrix**: Elastic skeleton (tissue structure)
//! 2. **Fluid Phase**: Pore fluid (interstitial fluid, blood)
//! 3. **Coupling**: Interaction between phases via drag force
//!
//! Biot's theory predicts two compressional waves:
//! - **Fast Wave (P1)**: In-phase motion of solid and fluid
//! - **Slow Wave (P2)**: Out-of-phase motion with high attenuation
//!
//! ## Literature References
//!
//! - Biot, M. A. (1956). "Theory of propagation of elastic waves in a
//!   fluid-saturated porous solid." *JASA*, 28(2), 168-178.
//! - Johnson, D. L., et al. (1987). "Theory of dynamic permeability and
//!   tortuosity in fluid-saturated porous media." *J. Fluid Mech*, 176, 379-402.
//! - Nguyen, V. H., et al. (2010). "Simulation of ultrasound propagation
//!   through bone using Biot theory." *IEEE UFFC*, 57(5), 1125-1131.
//! - Fellah, Z. E. A., & Depollier, C. (2000). "Transient acoustic wave
//!   propagation in rigid porous media." *JASA*, 107(2), 683-688.
//!
//! ## Applications
//!
//! - Bone acoustics
//! - Liver tissue characterization
//! - Lung parenchyma modeling
//! - Cartilage imaging
//! - Tumor microenvironment

pub mod biot;
pub mod properties;
pub mod waves;

pub mod material;
pub mod simulation;

#[cfg(test)]
mod tests;

pub use biot::BiotTheory;
pub use properties::PoroelasticProperties;
pub use waves::{WaveMode, WaveSpeeds};

pub use material::PoroelasticMaterial;
pub use simulation::PoroelasticSimulation;
