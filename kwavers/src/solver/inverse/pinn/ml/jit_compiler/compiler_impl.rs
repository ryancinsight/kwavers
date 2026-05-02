use crate::core::error::KwaversResult;
use crate::solver::inverse::pinn::ml::Geometry2D;
use std::sync::Arc;

use super::{
    CompiledKernel, CompilerStats, ExecutionPlan, JitCompiler, KernelData, MemoryLayout, ModelInfo,
    Operation, OptimizationLevel, VectorizationLevel,
};

impl JitCompiler {
    pub fn new(optimization_level: OptimizationLevel) -> Self {
        Self {
            kernel_cache: std::collections::HashMap::new(),
            optimization_level,
            cache_size_limit: 100,
            stats: CompilerStats {
                kernels_compiled: 0,
                cache_hit_rate: 1.0,
                avg_compile_time_ms: 0.0,
                avg_execution_time_us: 0.0,
                memory_usage: 0,
            },
        }
    }

    pub fn compile_pinn_model(
        &mut self,
        model: &dyn std::any::Any,
        geometry: &Geometry2D,
        kernel_name: &str,
    ) -> KwaversResult<CompiledKernel> {
        if let Some(cached_kernel) = self.kernel_cache.get(kernel_name) {
            return Ok(cached_kernel.clone());
        }

        let start_time = std::time::Instant::now();

        let model_info = self.analyze_model(model)?;
        let kernel_id = format!("{}_{}", kernel_name, model_info.hash);

        let execution_plan = self.generate_execution_plan(&model_info, geometry)?;

        let kernel = CompiledKernel {
            id: kernel_id.clone(),
            input_dims: model_info.input_dims.clone(),
            output_dims: model_info.output_dims.clone(),
            estimated_time_us: self.estimate_execution_time(&execution_plan),
            memory_required: self.estimate_memory_usage(&execution_plan),
            kernel_data: Arc::new(KernelData::Interpreted {
                model: Arc::new(()),
            }),
        };

        let compile_time = start_time.elapsed().as_millis() as f64;

        self.stats.kernels_compiled += 1;
        self.stats.avg_compile_time_ms = (self.stats.avg_compile_time_ms
            * (self.stats.kernels_compiled - 1) as f64
            + compile_time)
            / self.stats.kernels_compiled as f64;
        self.stats.memory_usage += kernel.memory_required;

        self.kernel_cache.insert(kernel_id, kernel.clone());
        self.enforce_cache_limit();

        Ok(kernel)
    }

    fn analyze_model(&self, _model: &dyn std::any::Any) -> KwaversResult<ModelInfo> {
        Ok(ModelInfo {
            input_dims: vec![3],
            output_dims: vec![1],
            num_layers: 4,
            _num_parameters: 10000,
            hash: "pinn_2d_wave_v1".to_string(),
            activation_functions: vec!["tanh".to_string(); 3],
        })
    }

    fn generate_execution_plan(
        &self,
        model_info: &ModelInfo,
        geometry: &Geometry2D,
    ) -> KwaversResult<ExecutionPlan> {
        let mut operations = Vec::new();

        operations.push(Operation::InputNormalization);

        for layer_idx in 0..model_info.num_layers {
            operations.push(Operation::LinearLayer {
                _input_size: if layer_idx == 0 { 3 } else { 200 },
                _output_size: if layer_idx == model_info.num_layers - 1 {
                    1
                } else {
                    200
                },
                _activation: model_info
                    .activation_functions
                    .get(layer_idx)
                    .cloned()
                    .unwrap_or_else(|| "linear".to_string()),
            });
        }

        operations.push(Operation::PhysicsConstraints {
            _geometry_type: geometry.geometry_type(),
            _wave_speed: 343.0,
        });

        Ok(ExecutionPlan {
            operations,
            _memory_layout: MemoryLayout::Contiguous,
            _vectorization: self.determine_vectorization_level(),
            _cache_optimization: true,
        })
    }

    pub(super) fn estimate_execution_time(&self, plan: &ExecutionPlan) -> f64 {
        let base_time_per_op = match self.optimization_level {
            OptimizationLevel::None => 50.0,
            OptimizationLevel::Basic => 10.0,
            OptimizationLevel::Aggressive => 2.0,
            OptimizationLevel::Maximum => 0.5,
        };

        plan.operations.len() as f64 * base_time_per_op
    }

    fn estimate_memory_usage(&self, _plan: &ExecutionPlan) -> usize {
        let bytes_per_param = match self.optimization_level {
            OptimizationLevel::None => 8,
            _ => 4,
        };

        200 * 200 * bytes_per_param
    }

    fn determine_vectorization_level(&self) -> VectorizationLevel {
        match self.optimization_level {
            OptimizationLevel::None => VectorizationLevel::None,
            OptimizationLevel::Basic => VectorizationLevel::Scalar,
            OptimizationLevel::Aggressive => VectorizationLevel::SIMD128,
            OptimizationLevel::Maximum => VectorizationLevel::SIMD256,
        }
    }

    fn enforce_cache_limit(&mut self) {
        while self.kernel_cache.len() > self.cache_size_limit {
            if let Some(oldest_key) = self.kernel_cache.keys().next().cloned() {
                if let Some(removed_kernel) = self.kernel_cache.remove(&oldest_key) {
                    self.stats.memory_usage = self
                        .stats
                        .memory_usage
                        .saturating_sub(removed_kernel.memory_required);
                }
            }
        }
    }

    pub fn get_stats(&self) -> &CompilerStats {
        &self.stats
    }

    pub fn clear_cache(&mut self) {
        self.kernel_cache.clear();
        self.stats.memory_usage = 0;
        self.stats.cache_hit_rate = 0.0;
    }
}
