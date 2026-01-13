//! Spectroscopic Unmixing Solvers

pub mod tikhonov;
pub mod unmixer;

pub use unmixer::SpectralUnmixer;
pub use tikhonov::tikhonov_solve;
