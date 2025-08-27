//! CUDA backend implementation
//!
//! This module provides NVIDIA CUDA acceleration with proper domain separation:
//! - `context`: Device context management
//! - `memory`: Memory allocation and transfers
//! - `kernels`: CUDA kernel source code
//! - `field_ops`: Field update operations
//! - `device`: Device detection and properties

pub mod context;
pub mod device;
pub mod field_ops;
pub mod kernels;
pub mod memory;

// Re-export main types
pub use context::CudaContext;
pub use device::{detect_cuda_devices, get_device_properties, DeviceProperties};
pub use kernels::CudaKernels;
pub use memory::{allocate_cuda_memory, device_to_host_cuda, host_to_device_cuda, CudaMemory};
