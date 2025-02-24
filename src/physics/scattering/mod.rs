// physics/scattering/mod.rs
pub mod acoustic;
pub mod optic;

pub use acoustic::{AcousticScatteringModel, compute_bubble_interactions, compute_mie_scattering, compute_rayleigh_scattering};
pub use optic::RayleighOpticalScatteringModel;