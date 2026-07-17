use super::{
    DataType, EdgeRuntime, EdgeRuntimePerformanceMonitor, ExecutionKernel, IOSpecification,
    MemoryAllocator,
};
use crate::inverse::pinn::ml::QuantizedModel;
use kwavers_core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;

impl EdgeRuntime {
    /// Create a new edge runtime for a specific hardware platform
    pub fn new(memory_limit_mb: usize) -> Self {
        let hardware_caps = Self::detect_hardware_capabilities();
        let total_memory = memory_limit_mb * 1024 * 1024;

        Self {
            model: None,
            allocator: MemoryAllocator::new(total_memory),
            kernel_cache: HashMap::new(),
            performance_monitor: EdgeRuntimePerformanceMonitor {
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
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn load_model(&mut self, model: QuantizedModel) -> KwaversResult<()> {
        self.validate_model_compatibility(&model)?;

        let model_memory = model.memory_usage();
        self.allocator.allocate_block(model_memory, 64)?;

        self.create_execution_kernels(&model)?;

        self.model = Some(model);
        Ok(())
    }

    /// Execute optimized inference
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn inference(&mut self, input: &[f32]) -> KwaversResult<Vec<f32>> {
        let start_time = std::time::Instant::now();

        let model = self.model.as_ref().ok_or_else(|| {
            KwaversError::System(kwavers_core::error::SystemError::ResourceUnavailable {
                resource: "No model loaded".to_string(),
            })
        })?;

        let input_size = std::mem::size_of_val(input);
        let output_size = model.metadata.compression_ratio as usize * 4;

        self.allocator.allocate_block(input_size, 16)?;
        self.allocator.allocate_block(output_size, 16)?;

        let mut current_input = input.to_vec();

        for kernel in self.kernel_cache.values() {
            let quantized_input = if self.hardware_caps.has_fpu {
                self.quantize_input_for_kernel(&current_input, kernel)?
            } else {
                self.fixed_point_inference(&current_input, kernel)?
            };

            current_input = self.execute_kernel(&quantized_input, kernel)?;
        }

        let output = self.dequantize_output(&current_input)?;

        let inference_time = start_time.elapsed().as_micros() as u64;
        self.update_performance_stats(inference_time);

        Ok(output)
    }
    /// Validate model compatibility.
    /// # Errors
    /// - Returns [`crate::KwaversError::System`] if the precondition for a System-class constraint is violated.
    ///
    pub(super) fn validate_model_compatibility(&self, model: &QuantizedModel) -> KwaversResult<()> {
        let model_memory = model.memory_usage();
        if model_memory > self.allocator.total_memory {
            return Err(KwaversError::System(
                kwavers_core::error::SystemError::ResourceUnavailable {
                    resource: format!(
                        "Model requires {} bytes, only {} available",
                        model_memory, self.allocator.total_memory
                    ),
                },
            ));
        }

        if let crate::inverse::pinn::ml::QuantizationScheme::None =
            &model.quantization_params.scheme
        {
            if !self.hardware_caps.has_fpu {
                return Err(KwaversError::System(
                    kwavers_core::error::SystemError::InvalidConfiguration {
                        parameter: "quantization".to_string(),
                        reason: "FP32 model requires FPU support".to_string(),
                    },
                ));
            }
        }

        let required_simd = match &model.quantization_params.scheme {
            crate::inverse::pinn::ml::QuantizationScheme::MixedPrecision {
                weight_bits, ..
            } => {
                if *weight_bits <= 8 {
                    8
                } else {
                    16
                }
            }
            _ => 8,
        };

        if self.hardware_caps.simd_width < required_simd {
            return Err(KwaversError::System(
                kwavers_core::error::SystemError::InvalidConfiguration {
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
    /// Create execution kernels.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn create_execution_kernels(&mut self, model: &QuantizedModel) -> KwaversResult<()> {
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
}
