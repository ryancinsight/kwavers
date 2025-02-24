// fft/mod.rs
pub mod fft3d;
pub mod ifft3d;
mod fft_core; // Make fft_core a module

pub use fft3d::Fft3d;
pub use ifft3d::Ifft3d;