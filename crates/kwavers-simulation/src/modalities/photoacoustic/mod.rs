//! Photoacoustic Imaging (PAI) Module
//!
//! Implements photoacoustic imaging physics for molecular and functional imaging.
//! Photoacoustic imaging combines optical excitation with acoustic detection to provide
//! high-resolution images with optical contrast and acoustic penetration depth.
//!
//! ## Physics Overview
//!
//! The photoacoustic effect describes the generation of acoustic waves through optical absorption:
//!
//! **Optical Absorption → Thermal Expansion → Acoustic Wave Generation**
//!
//! The photoacoustic wave equation couples optical fluence, thermal diffusion, and acoustic propagation:
//!
//! ```text
//! ∂²p/∂t² - c²∇²p = Γ μₐ Φ(r,t) ∂H/∂t
//! ```
//!
//! Where:
//! - `p`: Acoustic pressure (Pa)
//! - `c`: Speed of sound (m/s)
//! - `Γ`: Grüneisen parameter (thermoelastic efficiency) (dimensionless)
//! - `μₐ`: Optical absorption coefficient [m⁻¹]
//! - `Φ`: Optical fluence [W/m² or J/m²]
//! - `H`: Heating function [J/m³]
//!
//! ## Implementation Features
//!
//! - **Multi-wavelength simulation** for spectroscopic imaging
//! - **Heterogeneous tissue optical properties** (absorption, scattering, anisotropy)
//! - **GPU-accelerated wave propagation** via FDTD solver integration
//! - **Universal back-projection reconstruction** for image formation
//! - **Time-reversal algorithms** for initial pressure reconstruction
//! - **Real-time processing pipeline** with parallel wavelength computation
//!
//! ## Architecture
//!
//! This module follows Clean Architecture principles with clear layer separation:
//!
//! - **Domain Layer** (`types`): Domain models and type definitions
//! - **Application Layer** (`core`): Orchestration via [`PhotoacousticSimulator`]
//! - **Infrastructure Layer** (`optics`, `acoustics`, `reconstruction`): Technical implementations
//! - **Interface Layer** (`mod`): Public API and re-exports
//!
//! ## Module Structure
//!
//! - [`core`]: Core simulator struct ([`PhotoacousticSimulator`]) and orchestration logic
//! - [`optics`]: Optical fluence computation using diffusion approximation
//! - [`acoustics`]: Acoustic pressure generation and wave propagation
//! - [`reconstruction`]: Image reconstruction algorithms (UBP, time-reversal)
//! - [`types`]: Type definitions and re-exports from domain module
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
//! use kwavers_simulation::modalities::photoacoustic::PhotoacousticSimulator;
//! use kwavers_domain::grid::Grid;
//! use kwavers_domain::medium::homogeneous::HomogeneousMedium;
//! use kwavers_domain::imaging::photoacoustic::PhotoacousticParameters;
//!
//! # fn main() -> kwavers_core::error::KwaversResult<()> {
//! // Create computational grid
//! let grid = Grid::new(64, 64, 32, 0.001, 0.001, 0.001)?;
//!
//! // Define acoustic medium (SSOT: DENSITY_WATER_NOMINAL = 1000.0 kg/m³, SOUND_SPEED_WATER_SIM = 1500.0 m/s)
//! let medium = HomogeneousMedium::new(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, 0.5, 1.0, &grid);
//!
//! // Configure photoacoustic parameters
//! let parameters = PhotoacousticParameters::default();
//!
//! // Create simulator
//! let mut simulator = PhotoacousticSimulator::new(grid, parameters, &medium)?;
//!
//! // Compute optical fluence distribution
//! let fluence = simulator.compute_fluence()?;
//!
//! // Generate initial acoustic pressure
//! let initial_pressure = simulator.compute_initial_pressure(&fluence)?;
//!
//! // Run full simulation with wave propagation and reconstruction
//! let result = simulator.simulate(&initial_pressure)?;
//!
//! println!("SNR: {:.2} dB", result.snr);
//! println!("Reconstructed image: {:?}", result.reconstructed_image.dim());
//! # Ok(())
//! # }
//! ```
//!
//! ## Multi-Wavelength Spectroscopic Imaging
//!
//! ```rust,no_run
//! # use kwavers_simulation::modalities::photoacoustic::PhotoacousticSimulator;
//! # use kwavers_domain::grid::Grid;
//! # use kwavers_domain::medium::homogeneous::HomogeneousMedium;
//! # use kwavers_domain::imaging::photoacoustic::PhotoacousticParameters;
//! # use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
//! # fn main() -> kwavers_core::error::KwaversResult<()> {
//! # let grid = Grid::new(32, 32, 16, 0.001, 0.001, 0.001)?;
//! # let medium = HomogeneousMedium::new(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, 0.5, 1.0, &grid);
//! # let parameters = PhotoacousticParameters::default();
//! # let simulator = PhotoacousticSimulator::new(grid, parameters, &medium)?;
//! // Compute fluence for all wavelengths in parallel
//! let multi_wavelength_results = simulator.simulate_multi_wavelength()?;
//!
//! for (wavelength_idx, (fluence, pressure)) in multi_wavelength_results.iter().enumerate() {
//!     println!("Wavelength {}: max pressure = {:.2e} Pa", wavelength_idx, pressure.max_pressure);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Mathematical Specifications
//!
//! ### Photoacoustic Pressure Generation
//!
//! The initial pressure distribution is computed via the photoacoustic generation theorem:
//!
//! ```text
//! p₀(r) = Γ · μₐ(r) · Φ(r)
//! ```
//!
//! This relationship is fundamental to photoacoustic imaging and forms the basis for
//! quantitative imaging of optical absorption.
//!
//! ### Optical Fluence Computation
//!
//! Optical fluence is computed using the diffusion approximation to the radiative transfer equation:
//!
//! ```text
//! ∇·(D∇Φ) - μₐΦ = -S
//! ```
//!
//! Where:
//! - `D = 1/(3(μₐ + μₛ'))`: Diffusion coefficient (m)
//! - `μₛ'`: Reduced scattering coefficient [m⁻¹]
//! - `S`: Source term [W/m³]
//!
//! ### Universal Back-Projection Reconstruction
//!
//! The reconstruction algorithm implements spherical back-projection with distance weighting:
//!
//! ```text
//! p₀(r) = Σᵢ (1/|r - rᵢ|) · pᵢ(t = |r - rᵢ|/c)
//! ```
//!
//! Where the sum is over all detector positions `rᵢ`.
//!
//! ## References
//!
//! - Wang et al. (2009): "Photoacoustic tomography: in vivo imaging from organelles to organs"
//!   *Nature Methods* 6(1), 71-77. DOI: 10.1038/nmeth.1288
//! - Beard (2011): "Biomedical photoacoustic imaging"
//!   *Interface Focus* 1(4), 602-631. DOI: 10.1098/rsfs.2011.0028
//! - Treeby & Cox (2010): "k-Wave: MATLAB toolbox for the simulation and reconstruction of photoacoustic wave fields"
//!   *Journal of Biomedical Optics* 15(2), 021314. DOI: 10.1117/1.3360308
//! - Cox et al. (2007): "k-space propagation models for acoustically heterogeneous media"
//!   *The Journal of the Acoustical Society of America* 121(1), 168-173. DOI: 10.1121/1.2387816
//!
//! ## Design Patterns
//!
//! - **Facade Pattern**: [`PhotoacousticSimulator`] provides unified interface to complex subsystems
//! - **Strategy Pattern**: Multiple reconstruction algorithms (UBP, time-reversal) with common interface
//! - **Builder Pattern**: Implicit via configuration structs ([`PhotoacousticParameters`])
//! - **Template Method**: Simulation workflow with customizable steps (fluence → pressure → propagation → reconstruction)
//!
//! ## Performance Considerations
//!
//! - Parallel multi-wavelength computation using Rayon
//! - GPU acceleration for wave propagation via FDTD solver
//! - Efficient memory layout for 3D arrays (C-contiguous)
//! - Pre-allocated buffers for time-stepping loops
//!
//! ## Safety Invariants
//!
//! - All grid indices are bounds-checked during interpolation
//! - CFL condition enforced for numerical stability in wave propagation
//! - Non-negative pressure values guaranteed by physical constraints
//! - Finite field values validated throughout computation

pub mod acoustics;
pub mod core;
pub mod optics;
pub mod reconstruction;
pub mod types;

// Re-export main simulator
pub use core::PhotoacousticSimulator;

pub use kwavers_domain::imaging::photoacoustic::{
    InitialPressure, PhotoacousticOpticalProperties, PhotoacousticParameters, PhotoacousticResult,
};

// Re-export optical property data from domain
pub use kwavers_domain::medium::properties::OpticalPropertyData;

#[cfg(test)]
mod tests;
