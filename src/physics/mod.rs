// physics/mod.rs
pub mod mechanics;
pub mod optics;
pub mod thermodynamics;
pub mod chemistry;
pub mod scattering; // Consolidated scattering
pub mod heterogeneity;

pub use mechanics::acoustic_wave::NonlinearWave;
pub use mechanics::cavitation::CavitationModel;
pub use mechanics::streaming::StreamingModel;
pub use mechanics::viscosity::ViscosityModel;
pub use optics::diffusion::LightDiffusion;
pub use thermodynamics::heat_transfer::ThermalModel;
pub use chemistry::ChemicalModel;
pub use scattering::acoustic::AcousticScatteringModel;
pub use heterogeneity::HeterogeneityModel;