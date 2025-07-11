// physics/mechanics/acoustic_wave/mod.rs
pub mod nonlinear; // This will now refer to the new subdirectory

// Re-export NonlinearWave from the new structure.
// The actual struct will be defined in src/physics/mechanics/acoustic_wave/nonlinear/mod.rs
// and potentially composed of parts from other files in that directory.
// For now, ensure this line correctly points once the new nonlinear/mod.rs is set up.
// It might be initially commented out if NonlinearWave is not yet defined in the new mod.rs
pub use nonlinear::NonlinearWave;

pub mod viscoelastic_wave;
pub use viscoelastic_wave::ViscoelasticWave;
