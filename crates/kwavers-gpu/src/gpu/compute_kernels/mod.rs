//! GPU compute kernels for acoustic simulations.
//!
//! Provider-backed compute kernels for GPU acceleration. WGPU owns the current
//! WGSL implementations; additional providers implement the operation traits
//! only when they provide real kernels.

mod acoustic_field;
mod wave_equation_gpu;

pub use acoustic_field::{AcousticFieldKernel, AcousticFieldProvider, WgpuAcousticFieldProvider};
pub use wave_equation_gpu::WaveEquationGpu;
