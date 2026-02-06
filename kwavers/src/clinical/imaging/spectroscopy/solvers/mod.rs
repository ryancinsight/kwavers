//! Spectroscopic Unmixing Solvers

pub mod tikhonov;
pub mod unmixer;

pub use tikhonov::tikhonov_solve;
pub use unmixer::SpectralUnmixer;
