//! Sonoluminescence physics module
//! 
//! This module implements detailed physics for single-bubble sonoluminescence (SBSL)
//! and multi-bubble sonoluminescence (MBSL) based on scientific literature.
//! 
//! Key references:
//! - Brenner et al. (2002) "Single-bubble sonoluminescence" Rev. Mod. Phys. 74, 425
//! - Yasui (1997) "Alternative model of single-bubble sonoluminescence" Phys. Rev. E 56, 6750
//! - Hilgenfeldt et al. (1999) "Sonoluminescence light emission" Phys. Fluids 11, 1318
//! - Suslick & Flannigan (2008) "Inside a collapsing bubble" Annu. Rev. Phys. Chem. 59, 659

pub mod bubble_dynamics;
pub mod light_emission;
pub mod plasma_chemistry;
pub mod thermal_transport;
pub mod shock_dynamics;
pub mod spectral_analysis;
pub mod models;

// Re-export main types
pub use bubble_dynamics::{BubbleDynamics, RayleighPlessetSolver};
pub use light_emission::{LightEmissionModel, BlackbodyEmission, BremsstrahlingEmission};
pub use models::{SBSLModel, MBSLModel, SonoluminescenceParameters};
pub use spectral_analysis::{SpectralAnalyzer, EmissionSpectrum};