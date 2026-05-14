//! GPU Delay-and-Sum Beamforming Kernel
//!
//! High-performance GPU-accelerated delay-and-sum beamforming for volumetric ultrasound imaging.
//! Implements the core GPU compute pipeline with WGPU, including buffer management, parameter
//! passing, and result readback.
//!
//! # Performance Characteristics
//! - GPU acceleration: 10-100× speedup vs CPU
//! - Workgroup size: 8×8×8 threads
//! - Memory layout: Optimized for coalesced access
//!
//! # References
//! - Van Veen & Buckley (1988) "Beamforming: A versatile approach to spatial filtering"
//! - Jensen (1996) "Field: A Program for Simulating Ultrasound Systems"

mod params;
mod processor;
#[cfg(test)]
mod tests;

#[cfg(feature = "gpu")]
pub(super) use processor::dynamic_focus_dispatch::DynamicFocusGPU;
#[cfg(feature = "gpu")]
pub use processor::DelaySumGPU;
