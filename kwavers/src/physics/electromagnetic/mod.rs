//! Electromagnetic Physics Implementations
//!
//! This module provides concrete implementations of electromagnetic wave physics
//! across all frequency ranges, including:
//! - **Radiofrequency/Microwave**: Maxwell's equations solvers, antennas
//! - **Optical/Infrared**: Light diffusion, scattering, polarization (optics submodule)
//! - **Photoacoustic**: EM-acoustic coupling for imaging
//! - **Plasmonics**: Surface plasmon effects and nanophotonics
//!
//! ## Not yet implemented
//!
//! - **Time-domain Maxwell solver**: FDTD for ∂E/∂t = (∇×H − J)/ε, ∂H/∂t = −∇×E/μ
//!   to propagate sonoluminescence light.
//! - **PML absorbing boundaries**: Perfectly Matched Layer for open-domain optical problems.
//! - **Dispersive material models**: Frequency-dependent ε(ω), μ(ω) for broadband spectra.
//! - **Nonlinear optics**: Second- and third-order effects for intense pulses.
//! - **Photon transport coupling**: Radiative transfer equation linked to EM field.
//!
//! ## Hierarchical Organization
//!
//! Following physics principles, optics is a submodule of electromagnetics since
//! visible light is electromagnetic radiation in the 400-700nm wavelength range:
//!
//! ```text
//! Electromagnetic Physics
//! ├── Fundamental (equations/)     # Maxwell's equations, constitutive relations
//! ├── Optics (optics/)            # Visible light subset (400-700nm)
//! │   ├── Diffusion               # Radiative transfer, Monte Carlo
//! │   ├── Scattering              # Mie theory, Rayleigh scattering
//! │   ├── ElectromagneticPolarization            # Jones calculus
//! │   └── Sonoluminescence        # Light from cavitation
//! ├── Plasmonics (plasmonics/)    # Nanophotonic effects
//! ├── Photoacoustic (photoacoustic/) # EM-acoustic coupling
//! └── Solvers (solvers/)          # FDTD, FEM numerical methods
//! ```
//!
//! ## Architecture
//!
//! The electromagnetic physics implementations follow the physics specifications
//! in `physics::electromagnetic::equations` while delegating numerical methods to
//! the shared solver layer. Physics defines constitutive relations and material
//! properties; solvers provide numerical algorithms.

pub mod equations; // Electromagnetic wave equation specifications
                   // pub mod optics; // Moved to physics::optics
pub mod photoacoustic;
pub mod plasmonics;

// Re-export physics implementations
// Optics exports moved to physics::optics
#[allow(deprecated)]
pub use photoacoustic::{GrueneisenModel, GruneisenParameter, OpticalAbsorption};
pub use plasmonics::{MieTheory, NanoparticleArray, PlasmonicEnhancement};

// Re-export electromagnetic equation specifications for convenience
// Note: EMSource and related types are now in domain::source::electromagnetic
pub use equations::{
    EMFieldUtils, EMMaterialUtils, ElectromagneticWaveEquation, PhotoacousticCoupling,
};
