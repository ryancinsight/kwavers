// physics/scattering/mod.rs
pub mod acoustic;
pub mod optic;

pub use acoustic::{AcousticScattering, RayleighScattering, compute_bubble_interactions, compute_mie_scattering, compute_rayleigh_scattering};
pub use optic::RayleighOpticalScatteringModel;