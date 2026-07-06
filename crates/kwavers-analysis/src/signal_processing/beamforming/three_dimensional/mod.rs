//! Real-Time 3D Beamforming for GPU-Accelerated Volumetric Ultrasound Imaging
//!
//! This module implements high-performance 3D beamforming algorithms optimized for
//! volumetric ultrasound imaging with real-time processing capabilities. Extends
//! the 2D beamforming framework with provider-owned GPU acceleration.
//!
//! # Key Features
//! - **3D Delay-and-Sum Beamforming**: Full volumetric reconstruction with dynamic focusing
//! - **GPU Acceleration**: provider-owned kernels for 10-100× performance improvement
//! - **Real-Time Processing**: Streaming data pipeline with <10ms reconstruction time
//! - **4D Ultrasound Support**: 3D imaging with temporal dimension
//! - **Clinical Integration**: DICOM-compatible output with standard ultrasound formats
//!
//! # Performance Targets
//! - Reconstruction time: <10ms per volume
//! - Speedup: 10-100× vs CPU implementation
//! - Dynamic range: 30+ dB
//! - Memory efficiency: Streaming processing with minimal buffer overhead
//!
//! # Architecture
//! ```text
//! Raw RF Data -> Provider Buffer -> Beamforming Kernel -> Volume Reconstruction -> Post-Processing
//!     ↑              ↑              ↑                      ↑                    ↑
//!   Streaming     Memory       Provider Dispatch     3D Volume         Filtering &
//!   Acquisition   Management                       Interpolation      Enhancement
//! ```
//!
//! # Module Structure
//! - [`config`]: Configuration types, algorithm selection, and apodization windows
//! - `processor`: Provider-generic GPU processor initialization and setup
//! - `processing`: High-level processing interface (volume and streaming)
//! - `provider`: GPU provider contract implemented by `kwavers-gpu`
//! - `apodization`: Apodization weight generation for sidelobe reduction
//! - `steering`: Steering vector computation for adaptive beamforming
//! - `streaming`: Real-time streaming buffer management
//! - `metrics`: Performance metrics and memory usage tracking
//!
//! # References
//! - Van Veen & Buckley (1988) "Beamforming: A versatile approach to spatial filtering"
//! - Jensen (1996) "Field: A Program for Simulating Ultrasound Systems"
//! - Synnevåg et al. (2005) "Adaptive beamforming applied to medical ultrasound imaging"

mod apodization;
pub mod config;
pub(super) mod cpu;
mod metrics;
mod processing;
mod processor;
#[cfg(feature = "gpu")]
mod provider;
mod saft;
mod streaming;

// Public API re-exports
pub use config::{
    Beamforming3dApodizationWindow, BeamformingAlgorithm3D, BeamformingConfig3D, BeamformingMetrics,
};
pub use cpu::delay_and_sum_cpu as delay_and_sum_cpu_reference;
pub use processor::BeamformingProcessor3D;
#[cfg(feature = "gpu")]
pub use provider::BeamformingGpuProvider;
pub use saft::{SaftConfig, SaftProcessor};

#[cfg(test)]
mod tests;
