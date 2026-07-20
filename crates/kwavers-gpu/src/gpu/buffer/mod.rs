//! GPU buffer management
//!
//! This module provides safe, high-level abstractions for GPU buffer operations including:
//! - Buffer creation with automatic memory management
//! - Asynchronous data transfer between CPU and GPU
//! - Type-safe buffer operations using bytemuck traits
//! - Staging buffer patterns for GPU-to-CPU reads
//!
//! # Examples
//!
//! ```no_run
//! # use kwavers_gpu::gpu::buffer::{BufferUsage, GpuBufferData};
//! # async fn example(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), Box<dyn std::error::Error>> {
//! // Create a buffer with initial data
//! let data = vec![1.0f32, 2.0, 3.0, 4.0];
//! let buffer = GpuBufferData::create_with_data(
//!     device,
//!     &data,
//!     BufferUsage::STORAGE | BufferUsage::COPY_SRC
//! )?;
//!
//! // Read data back from GPU
//! let result: Vec<f32> = buffer.read_to_vec(device, queue).await?;
//! assert_eq!(result, data);
//! # Ok(())
//! # }
//! ```

mod gpu_buffer;
mod usage;

pub use gpu_buffer::GpuBufferData;
pub use usage::BufferUsage;
