//! PSTD solver extensions.
//!
//! Optional plugins that extend the canonical
//! [`crate::forward::pstd::PSTDSolver`] with capabilities beyond the
//! base linear acoustic-fluid (μ = 0) stress-velocity formulation.
//!
//! # Available extensions
//!
//! - [`elastic`] — adds isotropic elastic (μ ≥ 0) shear-stress and
//!   full-tensor velocity-divergence handling on top of PSTD's spectral
//!   stepper. The acoustic-fluid limit (μ = 0) reduces exactly to the
//!   baseline solver.
//!
//! # Architectural intent
//!
//! Before this layout the codebase had two separate pseudospectral wave
//! steppers — `pstd::PSTDSolver` (acoustic) and a parallel
//! `solver::forward::elastic_wave::ElasticWave::update_wave`
//! (`AcousticWaveModel` impl that hard-coded `μ = 0`, behaving as a second
//! acoustic solver). The duplicate lived in the elastic module tree because
//! it carried the spectral elastic primitives
//! (`update_{stress,velocity}_in_place`) needed for future elastic PSTD work.
//!
//! This `extensions/` tree consolidates the redundancy: the spectral elastic
//! primitives now live as a PSTD plugin, the duplicate `AcousticWaveModel`
//! impl is gone, and the canonical solver matrix in [`crate::forward`]
//! has one acoustic PSTD path with elastic capability switched on via this
//! plugin.

pub mod elastic;
pub mod elastic_orchestrator;
pub mod elastic_plugin;

pub use elastic::{
    PstdElasticPlugin, SpectralElasticConfig, SpectralStressUpdateInputs,
    SpectralVelocityUpdateInputs,
};
pub use elastic_orchestrator::{
    ElasticPml, ElasticPmlSpec, ElasticPstdMedium, ElasticPstdOrchestrator, ElasticPstdSensorData,
    ElasticPstdSourceMode, ElasticPstdVelocitySource,
};
pub use elastic_plugin::MechanicalStressPlugin;
