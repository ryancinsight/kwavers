//! Physics validation — analytical solutions and experimental benchmarks
//!
//! ## Contents
//!
//! - `gaussian_beam`: Gaussian beam radius measurement and parameters.
//! - `sonoluminescence_benchmarks`: Validates Minnaert resonance radius,
//!   Blake threshold, Rayleigh collapse time, Wien's law, and Planck spectrum
//!   against Brenner, Hilgenfeldt & Lohse (2002), Yasui (1997), and Putterman
//!   & Weninger (2000).

pub mod gaussian_beam;
pub mod sonoluminescence_benchmarks;

pub use gaussian_beam::{measure_beam_radius, GaussianBeamParameters};
pub use sonoluminescence_benchmarks::BrennerSBSLConditions;
