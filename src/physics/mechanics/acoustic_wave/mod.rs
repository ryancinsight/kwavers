// physics/mechanics/acoustic_wave/mod.rs
pub mod nonlinear; // This will now refer to the new subdirectory

// Re-export NonlinearWave from the new structure.
pub use nonlinear::{NonlinearWave, OptimizedNonlinearWave};

pub mod viscoelastic_wave;
pub use viscoelastic_wave::ViscoelasticWave;
