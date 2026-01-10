//! Contrast-Enhanced Ultrasound (CEUS) Physics
//!
//! Implements microbubble dynamics, nonlinear scattering, and perfusion modeling.
//!
//! # Components
//!
//! - **Microbubble Dynamics**: Radial oscillations, shell viscoelasticity
//! - **Nonlinear Scattering**: Harmonic generation
//! - **Perfusion Modeling**: Blood flow kinetics
//! - **Reconstruction**: Imaging algorithms

pub mod cloud_dynamics;
pub mod microbubble;
pub mod perfusion;
pub mod reconstruction;
pub mod scattering;

pub use cloud_dynamics::{
    CloudBubble, CloudConfig, CloudDynamics, CloudResponse, IncidentField, ScatteredField,
};
pub use microbubble::{BubbleDynamics, Microbubble, MicrobubblePopulation};
pub use perfusion::{FlowKinetics, PerfusionModel, TissueUptake};
pub use reconstruction::{CEUSReconstruction, ContrastImage};
pub use scattering::{HarmonicImaging, NonlinearScattering};

// Note: ContrastEnhancedUltrasound orchestrator has moved to crate::simulation::imaging::ceus
