//! Photoacoustic Imaging Module
//!
//! This module provides types and workflows for photoacoustic imaging, which combines
//! optical excitation with ultrasound detection to achieve deep-tissue molecular imaging.
//!
//! ## Architecture
//!
//! This module resides in the **clinical/imaging** layer because photoacoustic imaging
//! is an application-level workflow that combines multiple physics domains:
//! - Optical absorption and scattering (light propagation)
//! - Thermoelastic expansion (photoacoustic effect)
//! - Acoustic wave propagation (ultrasound detection)
//! - Image reconstruction (time-reversal, back-projection)
//!
//! ## Photoacoustic Effect
//!
//! The photoacoustic effect occurs when pulsed laser light is absorbed by tissue,
//! causing rapid thermoelastic expansion that generates acoustic waves:
//!
//! 1. **Optical Absorption**: Light energy is absorbed by chromophores
//! 2. **Temperature Rise**: Absorbed energy causes local heating
//! 3. **Thermoelastic Expansion**: Rapid expansion generates pressure waves
//! 4. **Acoustic Propagation**: Pressure waves propagate to detectors
//! 5. **Image Reconstruction**: Acoustic signals are reconstructed into images
//!
//! ### Governing Equations
//!
//! **Initial Pressure Distribution** (Grüneisen relaxation):
//! ```text
//! p₀(r) = Γ · μₐ(r) · Φ(r)
//! ```
//! where:
//! - p₀ = initial pressure (Pa)
//! - Γ = Grüneisen parameter (dimensionless, ~0.1-0.2 for tissue)
//! - μₐ = absorption coefficient (m⁻¹)
//! - Φ = optical fluence (J/m²)
//!
//! **Acoustic Wave Equation**:
//! ```text
//! ∇²p - (1/c²)∂²p/∂t² = 0, with p(r, t=0) = p₀(r)
//! ```
//!
//! ## Types Provided
//!
//! - **`PhotoacousticParameters`**: Configuration for PA simulation/imaging
//! - **`PhotoacousticResult`**: Output from PA workflows
//! - **`OpticalPropertyData`**: Tissue optical properties (μₐ, μₛ, g, n) from domain SSOT
//! - **`InitialPressure`**: Initial pressure distribution from optical absorption
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use kwavers::clinical::imaging::photoacoustic::{
//!     PhotoacousticParameters, OpticalProperties
//! };
//!
//! // Configure PA imaging parameters
//! let params = PhotoacousticParameters {
//!     wavelengths: vec![532.0, 700.0, 850.0], // Multi-spectral (nm)
//!     absorption_coefficients: vec![10.0, 5.0, 2.0], // μₐ (m⁻¹)
//!     gruneisen_parameters: vec![0.12, 0.12, 0.12], // Γ
//!     pulse_duration: 10e-9, // 10 ns laser pulse
//!     laser_fluence: 20.0, // 20 mJ/cm²
//!     speed_of_sound: 1540.0, // m/s (soft tissue)
//!     ..Default::default()
//! };
//!
//! // Get tissue optical properties (using new API)
//! // Use canonical domain SSOT types:
//! let blood_props = PhotoacousticOpticalProperties::blood(532.0); // Oxy-Hb peak -> OpticalPropertyData
//! let tissue_props = PhotoacousticOpticalProperties::soft_tissue(700.0); // Near-IR window -> OpticalPropertyData
//!
//! println!("Blood absorption at 532 nm: {} m⁻¹", blood_props.absorption_coefficient);
//! ```
//!
//! ## Migration Notice
//!
//! **⚠️ IMPORTANT**: This module was moved from `domain::imaging::photoacoustic` to
//! `clinical::imaging::photoacoustic` in Sprint 188 Phase 3 (Domain Layer Cleanup).
//!
//! ### Old Import (No Longer Valid)
//! ```rust,ignore
//! use crate::domain::imaging::photoacoustic::{PhotoacousticParameters, PhotoacousticResult};
//! ```
//!
//! ### New Import (Correct Location)
//! ```rust,ignore
//! use crate::clinical::imaging::photoacoustic::{PhotoacousticParameters, PhotoacousticResult};
//! ```
//!
//! ## Applications
//!
//! ### Medical Imaging
//! - **Vascular Imaging**: Blood vessel visualization (hemoglobin contrast)
//! - **Tumor Detection**: Enhanced absorption in malignant tissue
//! - **Lymph Node Mapping**: Sentinel node identification
//! - **Brain Imaging**: Functional neuroimaging (hemodynamic response)
//!
//! ### Pre-Clinical Research
//! - **Molecular Imaging**: Exogenous contrast agents
//! - **Drug Delivery**: Nanoparticle tracking
//! - **Tumor Microenvironment**: Oxygenation monitoring
//! - **Inflammation**: Macrophage infiltration
//!
//! ## References
//!
//! - Xu, M., & Wang, L. V. (2006). "Photoacoustic imaging in biomedicine."
//!   *Review of Scientific Instruments*, 77(4), 041101.
//! - Wang, L. V., & Hu, S. (2012). "Photoacoustic tomography: in vivo imaging
//!   from organelles to organs." *Science*, 335(6075), 1458-1462.
//! - Beard, P. (2011). "Biomedical photoacoustic imaging." *Interface Focus*, 1(4), 602-631.
//!
//! ## Safety Considerations
//!
//! ### Laser Safety (ANSI Z136.1)
//! - Maximum Permissible Exposure (MPE): 20 mJ/cm² at 532 nm
//! - Pulse duration limits: <50 ns for thermal confinement
//! - Eye protection: Required for operators (OD ≥4 at laser wavelength)
//!
//! ### Acoustic Safety (FDA/IEC)
//! - Mechanical Index (MI): <1.9 for diagnostic imaging
//! - Thermal Index (TI): <6 to prevent tissue heating
//! - Pressure limits: Peak negative <1 MPa (avoid cavitation)

pub mod types;

pub use types::{
    InitialPressure, PhotoacousticOpticalProperties, PhotoacousticParameters, PhotoacousticResult,
};
