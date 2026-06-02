//! Acoustic pressure generation and wave propagation for photoacoustic imaging.

pub mod pressure;
pub mod propagation;
#[cfg(test)]
mod tests;

pub use pressure::{compute_initial_pressure, compute_multi_wavelength_pressure};
pub use propagation::propagate_acoustic_wave;
