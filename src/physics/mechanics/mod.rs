// physics/mechanics/mod.rs
pub mod acoustic_wave;
pub mod cavitation;
pub mod streaming;
pub mod viscosity;
pub mod elastic_wave; // Added new module

pub use acoustic_wave::NonlinearWave;
pub use cavitation::CavitationModel;
pub use streaming::StreamingModel;
pub use viscosity::ViscosityModel;