// physics/mechanics/mod.rs

pub mod absorption;
pub mod acoustic_wave;
pub mod cavitation;
pub mod elastic_wave;
pub mod poroelastic; // Sprint 139: Poroelastic tissue with Biot theory
pub mod streaming;
pub mod viscosity;

pub use cavitation::CavitationModel;
pub use elastic_wave::ElasticWave;
pub use streaming::StreamingModel;
pub use viscosity::ViscosityModel;
