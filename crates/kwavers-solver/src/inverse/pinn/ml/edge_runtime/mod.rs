//! Edge Deployment Runtime for PINN Models
//!
//! This module provides optimized runtime execution for quantized PINN models
//! on resource-constrained edge devices including ARM, RISC-V, and embedded systems.

use std::collections::HashMap;

mod hardware;
mod kernel;
mod memory;
mod monitoring;
mod runtime;
#[cfg(test)]
mod tests;

/// Edge deployment runtime
#[derive(Debug)]
pub struct EdgeRuntime {
    pub(super) model: Option<crate::inverse::pinn::ml::QuantizedModel>,
    pub(super) allocator: MemoryAllocator,
    pub(super) kernel_cache: HashMap<String, ExecutionKernel>,
    pub(super) performance_monitor: EdgeRuntimePerformanceMonitor,
    pub(super) hardware_caps: HardwareCapabilities,
}

/// Memory allocator for constrained environments
#[derive(Debug)]
pub struct MemoryAllocator {
    pub(super) total_memory: usize,
    pub(super) allocations: Vec<MemoryBlock>,
    pub(super) fragmentation_ratio: f32,
}

/// Memory block allocation
#[derive(Debug, Clone)]
pub(super) struct MemoryBlock {
    pub start_address: usize,
    pub size: usize,
    pub allocated: bool,
    pub _alignment: usize,
}

/// Execution kernel for optimized inference
#[derive(Debug, Clone)]
pub struct ExecutionKernel {
    pub id: String,
    pub io_spec: IOSpecification,
    pub estimated_time_us: f64,
    pub memory_required: usize,
}

/// Input/Output specification
#[derive(Debug, Clone)]
pub struct IOSpecification {
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub input_dtype: DataType,
    pub output_dtype: DataType,
}

/// Data type specifications for edge devices
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Float32,
    Float16,
    Int8,
    Int4,
}

/// Hardware capabilities detection
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    pub architecture: Architecture,
    pub instruction_sets: Vec<String>,
    pub total_memory_mb: usize,
    pub has_fpu: bool,
    pub simd_width: usize,
    pub cache_line_size: usize,
}

/// CPU architecture types
#[derive(Debug, Clone)]
pub enum Architecture {
    ARM,
    ARM64,
    RISCV,
    X86,
    X86_64,
    Other(String),
}

/// Performance monitoring for edge devices
#[derive(Debug, Clone)]
pub struct EdgeRuntimePerformanceMonitor {
    pub inference_count: u64,
    pub total_inference_time_us: u64,
    pub peak_memory_usage: usize,
    pub avg_latency_us: f64,
    pub memory_efficiency: f32,
}
