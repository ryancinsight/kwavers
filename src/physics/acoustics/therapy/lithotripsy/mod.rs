//! Lithotripsy Physics Module
//!
//! Comprehensive simulation suite for ultrasound stone fragmentation including:
//! - Stone material properties and fracture mechanics
//! - Shock wave generation and propagation
//! - Cavitation cloud dynamics for stone erosion
//! - Bioeffects modeling for surrounding tissue safety
//! - Shared components with sonodynamic therapy and histotripsy
//!
//! ## Physics Overview
//!
//! Lithotripsy combines nonlinear acoustics, cavitation physics, and material fracture
//! mechanics to fragment urinary calculi (kidney stones) using focused ultrasound.
//!
//! ## Key Components
//!
//! 1. **Stone Fracture Mechanics**: Griffith criterion, dynamic fracture propagation
//! 2. **Nonlinear Wave Propagation**: KZK equation for shock wave formation
//! 3. **Cavitation Cloud Dynamics**: Bubble cloud expansion and collapse
//! 4. **Bioeffects**: Tissue damage assessment and safety monitoring
//!
//! ## Shared Components
//!
//! Components shared with sonodynamic therapy and histotripsy:
//! - Nonlinear acoustic propagation (KZK solver)
//! - Cavitation physics (bubble dynamics, cloud formation)
//! - Bioeffects modeling (thermal and mechanical damage)
//! - Safety monitoring and treatment planning

pub mod bioeffects;
pub mod cavitation_cloud;
pub mod shock_wave;
pub mod stone_fracture;

pub use bioeffects::{BioeffectsModel, SafetyAssessment, TissueDamageAssessment};
pub use cavitation_cloud::{CavitationCloudDynamics, CloudParameters};
pub use shock_wave::{ShockWaveGenerator, ShockWavePropagation};
pub use stone_fracture::{FractureMechanics, StoneFractureModel, StoneMaterial};
