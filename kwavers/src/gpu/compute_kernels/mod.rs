//! GPU compute kernels for acoustic simulations.
//!
//! High-performance compute shaders using wgpu for cross-platform GPU acceleration.
//! Supports both integrated and discrete GPUs with automatic fallback.

mod acoustic_field;
mod wave_equation_gpu;

pub use acoustic_field::AcousticFieldKernel;
pub use wave_equation_gpu::WaveEquationGpu;
