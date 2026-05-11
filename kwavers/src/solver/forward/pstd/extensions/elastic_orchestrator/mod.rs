//! ElasticPSTD orchestrator — minimal viable propagate driver around
//! [`super::PstdElasticPlugin`].
//!
//! # Scope
//!
//! Drives a homogeneous- or heterogeneous-medium isotropic elastic
//! pseudospectral simulation with leapfrog stress-velocity time stepping,
//! optional additive velocity-source injection at masked grid points, and
//! optional sensor recording of `(vx, vy, vz)` traces. PML, body forces,
//! viscoelastic damping, and anisotropic stiffness are deliberately
//! out-of-scope at this layer.
//!
//! # Theorem (acoustic-fluid limit, executable)
//!
//! For the same grid, time step, and source signal, running this
//! orchestrator with `μ ≡ 0` reproduces a baseline acoustic
//! pseudospectral solver to numerical precision modulo source-injection
//! semantics. The unit test
//! `pstd_elastic_plugin_reduces_to_acoustic_when_mu_is_zero` in
//! `physics/acoustics/mechanics/elastic_wave/tests.rs` asserts the
//! theorem in the spectral domain.

mod orchestrator;
mod types;

#[cfg(test)]
mod tests;

pub use orchestrator::ElasticPstdOrchestrator;
pub use types::{
    ElasticPstdMedium, ElasticPstdSensorData, ElasticPstdSourceMode, ElasticPstdVelocitySource,
};
