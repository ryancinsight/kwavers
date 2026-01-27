//! Edge Deployment Runtime for PINN Models
//!
//! This module provides optimized runtime execution for quantized PINN models
//! on resource-constrained edge devices including ARM, RISC-V, and embedded systems.

use crate::core::error::{KwaversError, KwaversResult};
use crate::solver::inverse::pinn::ml::quantization::LayerInfo;
use crate::solver::inverse::pinn::ml::QuantizedModel;
use std::collections::HashMap;

/// Edge deployment runtime
#[derive(Debug)]
pub struct EdgeRuntime {
    /// Loaded quantized model
    model: Option<QuantizedModel>,
    /// Memory allocator for constrained environments
    allocator: MemoryAllocator,
    /// Execution kernel cache
    kernel_cache: HashMap<String, ExecutionKernel>,
    /// Performance monitoring
    performance_monitor: PerformanceMonitor,
    /// Hardware capabilities
    hardware_caps: HardwareCapabilities,
}

/// Memory allocator for constrained environments
#[derive(Debug)]
pub struct MemoryAllocator {
    /// Total available memory (bytes)
    total_memory: usize,
    /// Allocated memory blocks
    allocations: Vec<MemoryBlock>,
    /// Memory fragmentation tracking
    fragmentation_ratio: f32,
}

/// Memory block allocation
#[derive(Debug, Clone)]
struct MemoryBlock {
    pub start_address: usize,
    pub size: usize,
    pub allocated: bool,
    pub _alignment: usize,
}

/// Execution kernel for optimized inference
#[derive(Debug, Clone)]
pub struct ExecutionKernel {
    /// Kernel identifier
    pub id: String,
    /// Input/output specifications
    pub io_spec: IOSpecification,
    /// Estimated execution time (microseconds)
    pub estimated_time_us: f64,
    /// Memory requirements
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
    /// CPU architecture
    pub architecture: Architecture,
    /// Available instruction sets
    pub instruction_sets: Vec<String>,
    /// Total memory (MB)
    pub total_memory_mb: usize,
    /// Has floating point unit
    pub has_fpu: bool,
    /// SIMD capabilities
    pub simd_width: usize,
    /// Cache line size
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
pub struct PerformanceMonitor {
    /// Inference call count
    pub inference_count: u64,
    /// Total inference time (microseconds)
    pub total_inference_time_us: u64,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: usize,
    /// Average latency (microseconds)
    pub avg_latency_us: f64,
    /// Memory efficiency ratio
    pub memory_efficiency: f32,
}

impl EdgeRuntime {
    /// Create a new edge runtime for a specific hardware platform
    pub fn new(memory_limit_mb: usize) -> Self {
        let hardware_caps = Self::detect_hardware_capabilities();
        let total_memory = memory_limit_mb * 1024 * 1024;

        Self {
            model: None,
            allocator: MemoryAllocator::new(total_memory),
            kernel_cache: HashMap::new(),
            performance_monitor: PerformanceMonitor {
                inference_count: 0,
                total_inference_time_us: 0,
                peak_memory_usage: 0,
                avg_latency_us: 0.0,
                memory_efficiency: 1.0,
            },
            hardware_caps,
        }
    }

    /// Load a quantized model for edge deployment
    pub fn load_model(&mut self, model: QuantizedModel) -> KwaversResult<()> {
        // Validate model compatibility with hardware
        self.validate_model_compatibility(&model)?;

        // Allocate memory for model weights
        let model_memory = model.memory_usage();
        self.allocator.allocate_block(model_memory, 64)?; // 64-byte alignment

        // Create execution kernels for each layer
        self.create_execution_kernels(&model)?;

        self.model = Some(model);
        Ok(())
    }

    /// Execute optimized inference
    pub fn inference(&mut self, input: &[f32]) -> KwaversResult<Vec<f32>> {
        let start_time = std::time::Instant::now();

        let model = self.model.as_ref().ok_or_else(|| {
            KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                resource: "No model loaded".to_string(),
            })
        })?;

        // Allocate input/output buffers
        let input_size = std::mem::size_of_val(input);
        let output_size = model.metadata.compression_ratio as usize * 4; // Estimate

        self.allocator.allocate_block(input_size, 16)?;
        self.allocator.allocate_block(output_size, 16)?;

        // Execute inference pipeline
        let mut current_input = input.to_vec();

        for kernel in self.kernel_cache.values() {
            // Apply quantization to input if needed
            let quantized_input = if self.hardware_caps.has_fpu {
                self.quantize_input_for_kernel(&current_input, kernel)?
            } else {
                // Fixed-point processing for devices without FPU
                self.fixed_point_inference(&current_input, kernel)?
            };

            // Execute kernel (simulated optimized execution)
            current_input = self.execute_kernel(&quantized_input, kernel)?;
        }

        // Dequantize final output
        let output = self.dequantize_output(&current_input)?;

        // Update performance statistics
        let inference_time = start_time.elapsed().as_micros() as u64;
        self.update_performance_stats(inference_time);

        Ok(output)
    }

    /// Validate model compatibility with hardware
    fn validate_model_compatibility(&self, model: &QuantizedModel) -> KwaversResult<()> {
        // Check memory requirements
        let model_memory = model.memory_usage();
        if model_memory > self.allocator.total_memory {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
                    resource: format!(
                        "Model requires {} bytes, only {} available",
                        model_memory, self.allocator.total_memory
                    ),
                },
            ));
        }

        // Check quantization compatibility
        match &model.quantization_params.scheme {
            crate::solver::inverse::pinn::ml::QuantizationScheme::None => {
                // FP32 requires FPU
                if !self.hardware_caps.has_fpu {
                    return Err(KwaversError::System(
                        crate::core::error::SystemError::InvalidConfiguration {
                            parameter: "quantization".to_string(),
                            reason: "FP32 model requires FPU support".to_string(),
                        },
                    ));
                }
            }
            _ => {
                // Quantized models are compatible with most edge devices
            }
        }

        // Check SIMD requirements
        let required_simd = match &model.quantization_params.scheme {
            crate::solver::inverse::pinn::ml::QuantizationScheme::MixedPrecision {
                weight_bits,
                ..
            } => {
                if *weight_bits <= 8 {
                    8
                } else {
                    16
                }
            }
            _ => 8, // Default 8-bit operations
        };

        if self.hardware_caps.simd_width < required_simd {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "simd_width".to_string(),
                    reason: format!(
                        "Model requires {} SIMD width, hardware provides {}",
                        required_simd, self.hardware_caps.simd_width
                    ),
                },
            ));
        }

        Ok(())
    }

    /// Create optimized execution kernels
    fn create_execution_kernels(&mut self, model: &QuantizedModel) -> KwaversResult<()> {
        for (layer_idx, layer) in model.original_layers.iter().enumerate() {
            let kernel = ExecutionKernel {
                id: format!("layer_{}", layer_idx),
                io_spec: IOSpecification {
                    input_shape: vec![layer.input_size],
                    output_shape: vec![layer.output_size],
                    input_dtype: DataType::Float32,
                    output_dtype: DataType::Float32,
                },
                estimated_time_us: self.estimate_kernel_time(layer),
                memory_required: self.estimate_kernel_memory(layer),
            };

            self.kernel_cache.insert(kernel.id.clone(), kernel);
        }

        Ok(())
    }

    /// Estimate kernel execution time
    fn estimate_kernel_time(&self, layer: &LayerInfo) -> f64 {
        let operations = layer.input_size * layer.output_size;

        // Base time per operation (microseconds)
        let base_time_per_op = match self.hardware_caps.architecture {
            Architecture::ARM | Architecture::ARM64 => 0.01, // Faster mobile CPUs
            Architecture::RISCV => 0.05,                     // Slower embedded CPUs
            Architecture::X86 | Architecture::X86_64 => 0.005, // Desktop CPUs
            Architecture::Other(_) => 0.02,
        };

        // Adjust for SIMD capabilities
        let simd_factor = self.hardware_caps.simd_width as f64 / 8.0;
        let adjusted_time = base_time_per_op / simd_factor;

        operations as f64 * adjusted_time
    }

    /// Estimate kernel memory requirements
    fn estimate_kernel_memory(&self, layer: &LayerInfo) -> usize {
        // Weight matrix + bias + intermediate buffers
        let weight_memory = layer.input_size * layer.output_size * std::mem::size_of::<i8>();
        let bias_memory = layer.output_size * std::mem::size_of::<f32>();
        let intermediate_memory = layer.output_size * std::mem::size_of::<f32>() * 2; // Input + output

        weight_memory + bias_memory + intermediate_memory
    }

    /// Quantize input for kernel execution
    fn quantize_input_for_kernel(
        &self,
        input: &[f32],
        kernel: &ExecutionKernel,
    ) -> KwaversResult<Vec<f32>> {
        // Apply hardware-specific quantization
        match self.hardware_caps.architecture {
            Architecture::ARM | Architecture::ARM64 => {
                // Use NEON instructions for quantization
                self.neon_quantize(input, kernel)
            }
            Architecture::RISCV => {
                // Use RISC-V vector extensions
                self.riscv_quantize(input, kernel)
            }
            _ => {
                // Fallback software quantization
                self.software_quantize(input, kernel)
            }
        }
    }

    /// NEON-optimized quantization (ARM)
    fn neon_quantize(&self, input: &[f32], kernel: &ExecutionKernel) -> KwaversResult<Vec<f32>> {
        // In practice, this would use ARM NEON intrinsics
        // For now, simulate optimized quantization
        let _ = kernel; // Explicitly mark as used if not currently in implementation
        Ok(input.iter().map(|&x| x.clamp(-1.0, 1.0)).collect())
    }

    /// RISC-V optimized quantization
    fn riscv_quantize(&self, input: &[f32], kernel: &ExecutionKernel) -> KwaversResult<Vec<f32>> {
        // In practice, this would use RISC-V vector instructions
        // For now, simulate optimized quantization
        let _ = kernel; // Explicitly mark as used if not currently in implementation
        Ok(input.iter().map(|&x| x.clamp(-1.0, 1.0)).collect())
    }

    /// Software fallback quantization
    fn software_quantize(
        &self,
        input: &[f32],
        kernel: &ExecutionKernel,
    ) -> KwaversResult<Vec<f32>> {
        match kernel.io_spec.input_dtype {
            DataType::Float32 => Ok(input.to_vec()),
            DataType::Float16 => {
                // Simulate FP16 conversion
                Ok(input.to_vec())
            }
            DataType::Int8 => {
                // Simulate 8-bit quantization
                let scale = input.iter().map(|x| x.abs()).fold(0.0, f32::max) / 127.0;
                Ok(input
                    .iter()
                    .map(|&x| (x / scale).clamp(-127.0, 127.0))
                    .collect())
            }
            DataType::Int4 => {
                // Simulate 4-bit quantization
                let scale = input.iter().map(|x| x.abs()).fold(0.0, f32::max) / 7.0;
                Ok(input
                    .iter()
                    .map(|&x| (x / scale).clamp(-7.0, 7.0))
                    .collect())
            }
        }
    }

    /// Fixed-point inference for devices without FPU
    fn fixed_point_inference(
        &self,
        input: &[f32],
        kernel: &ExecutionKernel,
    ) -> KwaversResult<Vec<f32>> {
        let _ = kernel;
        // Convert to fixed-point representation
        let fixed_input: Vec<i32> = input
            .iter()
            .map(|&x| (x * 65536.0) as i32) // 16.16 fixed point
            .collect();

        // Simulate fixed-point computation
        let output: Vec<f32> = fixed_input.iter().map(|&x| x as f32 / 65536.0).collect();

        Ok(output)
    }

    /// Execute a single kernel
    fn execute_kernel(&self, input: &[f32], kernel: &ExecutionKernel) -> KwaversResult<Vec<f32>> {
        // Simulate optimized kernel execution
        // In practice, this would call JIT-compiled or hand-optimized kernels

        let mut output = vec![0.0; kernel.io_spec.output_shape[0]];

        let Some(ref quantized_model) = self.model else {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "model".to_string(),
                    reason: "No quantized model loaded".to_string(),
                },
            ));
        };

        let layer_idx = kernel
            .id
            .strip_prefix("layer_")
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or_else(|| {
                KwaversError::System(crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "kernel.id".to_string(),
                    reason: format!("Invalid kernel id: {}", kernel.id),
                })
            })?;

        let layer = quantized_model
            .original_layers
            .get(layer_idx)
            .ok_or_else(|| {
                KwaversError::System(crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "model.layers".to_string(),
                    reason: format!("Missing layer index {}", layer_idx),
                })
            })?;

        let weight_tensor = quantized_model
            .quantized_weights
            .get(layer_idx * 2)
            .ok_or_else(|| {
                KwaversError::System(crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "model.quantized_weights".to_string(),
                    reason: format!("Missing weights for layer {}", layer_idx),
                })
            })?;
        let bias_tensor = quantized_model
            .quantized_weights
            .get(layer_idx * 2 + 1)
            .ok_or_else(|| {
                KwaversError::System(crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "model.quantized_weights".to_string(),
                    reason: format!("Missing biases for layer {}", layer_idx),
                })
            })?;

        let input_len = input.len().min(layer.input_size);
        let output_len = output.len().min(layer.output_size);
        output.truncate(output_len);

        // Dequantize weights and biases for execution
        // In a real optimized runtime, we would use quantized kernels
        let weights = weight_tensor.dequantize();
        let biases = bias_tensor.dequantize();

        for out_idx in 0..output.len() {
            let mut sum = 0.0f32;
            for (j, &input_val) in input.iter().enumerate().take(input_len) {
                let weight_index = j * layer.output_size + out_idx;
                if weight_index < weights.len() {
                    sum += input_val * weights[weight_index];
                }
            }

            if out_idx < biases.len() {
                sum += biases[out_idx];
            }

            output[out_idx] = sum.tanh();
        }

        Ok(output)
    }

    /// Dequantize output
    fn dequantize_output(&self, quantized_output: &[f32]) -> KwaversResult<Vec<f32>> {
        // Apply hardware-specific dequantization
        match self.hardware_caps.architecture {
            Architecture::ARM | Architecture::ARM64 => self.neon_dequantize(quantized_output),
            Architecture::RISCV => self.riscv_dequantize(quantized_output),
            _ => Ok(quantized_output.to_vec()),
        }
    }

    /// NEON dequantization
    fn neon_dequantize(&self, input: &[f32]) -> KwaversResult<Vec<f32>> {
        // Simulate NEON dequantization
        Ok(input.to_vec())
    }

    /// RISC-V dequantization
    fn riscv_dequantize(&self, input: &[f32]) -> KwaversResult<Vec<f32>> {
        // Simulate RISC-V dequantization
        Ok(input.to_vec())
    }

    /// Update performance statistics
    fn update_performance_stats(&mut self, inference_time_us: u64) {
        self.performance_monitor.inference_count += 1;
        self.performance_monitor.total_inference_time_us += inference_time_us;

        self.performance_monitor.avg_latency_us = self.performance_monitor.total_inference_time_us
            as f64
            / self.performance_monitor.inference_count as f64;

        // Update memory efficiency
        let current_memory = self.allocator.get_allocated_memory();
        if current_memory > self.performance_monitor.peak_memory_usage {
            self.performance_monitor.peak_memory_usage = current_memory;
        }

        self.performance_monitor.memory_efficiency =
            current_memory as f32 / self.allocator.total_memory as f32;
    }

    /// Detect hardware capabilities
    /// Detect hardware capabilities using target_arch and system info
    fn detect_hardware_capabilities() -> HardwareCapabilities {
        let architecture = if cfg!(target_arch = "aarch64") {
            Architecture::ARM64
        } else if cfg!(target_arch = "arm") {
            Architecture::ARM
        } else if cfg!(target_arch = "riscv64") {
            Architecture::RISCV
        } else if cfg!(target_arch = "x86") {
            Architecture::X86
        } else if cfg!(target_arch = "x86_64") {
            Architecture::X86_64
        } else {
            Architecture::Other(std::env::consts::ARCH.to_string())
        };

        let mut instruction_sets = Vec::new();
        if cfg!(target_feature = "neon") {
            instruction_sets.push("NEON".to_string());
        }
        if cfg!(target_feature = "sse") {
            instruction_sets.push("SSE".to_string());
        }
        if cfg!(target_feature = "avx") {
            instruction_sets.push("AVX".to_string());
        }
        if cfg!(target_feature = "avx2") {
            instruction_sets.push("AVX2".to_string());
        }

        HardwareCapabilities {
            architecture,
            instruction_sets,
            total_memory_mb: 512, // Default for embedded
            has_fpu: cfg!(target_arch = "x86_64")
                || cfg!(target_arch = "aarch64")
                || cfg!(target_feature = "neon")
                || cfg!(target_feature = "sse2"),
            simd_width: if cfg!(target_feature = "avx2") {
                256
            } else if cfg!(target_feature = "neon") || cfg!(target_feature = "sse") {
                128
            } else {
                64
            },
            cache_line_size: 64,
        }
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> &PerformanceMonitor {
        &self.performance_monitor
    }

    /// Get hardware capabilities
    pub fn get_hardware_caps(&self) -> &HardwareCapabilities {
        &self.hardware_caps
    }
}

impl MemoryAllocator {
    /// Create a new memory allocator
    pub fn new(total_memory: usize) -> Self {
        Self {
            total_memory,
            allocations: Vec::new(),
            fragmentation_ratio: 0.0,
        }
    }

    /// Allocate a memory block with alignment
    pub fn allocate_block(&mut self, size: usize, alignment: usize) -> KwaversResult<usize> {
        let aligned_size = size.div_ceil(alignment) * alignment;

        // Check if we have enough total memory
        let used_memory: usize = self.allocations.iter().map(|b| b.size).sum();
        if used_memory + aligned_size > self.total_memory {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
                    resource: format!(
                        "Memory limit reached: {}/{} bytes",
                        used_memory + aligned_size,
                        self.total_memory
                    ),
                },
            ));
        }

        // Find a gap between existing allocations or append at the end
        let mut best_start = 0;
        let mut allocations = self.allocations.clone();
        allocations.sort_by_key(|b| b.start_address);

        for block in &allocations {
            if block.start_address >= best_start + aligned_size {
                // Found a gap
                break;
            }
            best_start = (block.start_address + block.size).div_ceil(alignment) * alignment;
        }

        if best_start + aligned_size > self.total_memory {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
                    resource: "Memory fragmentation: No contiguous block found".to_string(),
                },
            ));
        }

        let new_block = MemoryBlock {
            start_address: best_start,
            size: aligned_size,
            allocated: true,
            _alignment: alignment,
        };

        self.allocations.push(new_block);
        self.update_fragmentation_stats();

        Ok(best_start)
    }

    fn update_fragmentation_stats(&mut self) {
        if self.allocations.is_empty() {
            self.fragmentation_ratio = 0.0;
            return;
        }

        let total_allocated: usize = self.allocations.iter().map(|b| b.size).sum();
        let max_address = self
            .allocations
            .iter()
            .map(|b| b.start_address + b.size)
            .max()
            .unwrap_or(0);

        if max_address == 0 {
            self.fragmentation_ratio = 0.0;
        } else {
            self.fragmentation_ratio = 1.0 - (total_allocated as f32 / max_address as f32);
        }
    }

    /// Get total allocated memory
    pub fn get_allocated_memory(&self) -> usize {
        self.allocations
            .iter()
            .filter(|block| block.allocated)
            .map(|block| block.size)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_runtime_creation() {
        let runtime = EdgeRuntime::new(64); // 64MB limit
        assert_eq!(runtime.allocator.total_memory, 64 * 1024 * 1024);
        assert!(runtime.model.is_none());
    }

    #[test]
    fn test_memory_allocator() {
        let mut allocator = MemoryAllocator::new(1024 * 1024); // 1MB

        // Allocate a block
        let address = allocator.allocate_block(1024, 64).unwrap();
        assert_eq!(address, 0);

        // Check allocated memory
        assert_eq!(allocator.get_allocated_memory(), 1024);
    }

    #[test]
    fn test_hardware_capabilities() {
        let caps = EdgeRuntime::detect_hardware_capabilities();

        // Should detect architecture and capabilities based on current platform
        match caps.architecture {
            Architecture::ARM64 | Architecture::ARM => {
                assert!(caps.has_fpu, "ARM architectures should have FPU");
            }
            Architecture::X86_64 | Architecture::X86 => {
                assert!(caps.has_fpu, "x86 architectures should have FPU");
            }
            Architecture::RISCV | Architecture::Other(_) => {
                // Other architectures may have different capabilities
            }
        }

        assert!(
            caps.simd_width >= 64 || matches!(caps.architecture, Architecture::Other(_)),
            "Should have at least basic SIMD or be other architecture"
        );
        assert!(caps.total_memory_mb > 0, "Should detect non-zero memory");
    }

    #[test]
    fn test_data_type_quantization() {
        let runtime = EdgeRuntime::new(64);
        let input = vec![1.0, -1.0, 0.5, -0.5];
        let kernel = ExecutionKernel {
            id: "test".to_string(),
            io_spec: IOSpecification {
                input_shape: vec![4],
                output_shape: vec![4],
                input_dtype: DataType::Int8,
                output_dtype: DataType::Float32,
            },
            estimated_time_us: 100.0,
            memory_required: 1024,
        };

        let result = runtime.software_quantize(&input, &kernel);
        assert!(result.is_ok());

        let quantized = result.unwrap();
        assert_eq!(quantized.len(), 4);

        // Check that values are clamped to expected range
        for &val in &quantized {
            assert!((-127.0f32..=127.0f32).contains(&val));
        }
    }
}
