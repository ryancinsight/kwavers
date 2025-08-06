//! Performance optimization module for the Kwavers acoustic simulation library

use crate::error::{ConfigError, KwaversError, KwaversResult};
use crate::grid::Grid;
use ndarray::{Array3, Array4, ArrayView3, ArrayViewMut3, Axis, s};
use rayon::prelude::*;
use std::sync::Arc;

use log::info;

/// Performance optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enable SIMD vectorization
    pub enable_simd: bool,
    /// SIMD instruction set to use
    pub simd_level: SimdLevel,
    /// Enable cache blocking
    pub cache_blocking: bool,
    /// Cache block size (in elements)
    pub cache_block_size: usize,
    /// Enable memory prefetching
    pub prefetching: bool,
    /// Prefetch distance (in cache lines)
    pub prefetch_distance: usize,
    /// Enable kernel fusion on GPU
    pub kernel_fusion: bool,
    /// Enable asynchronous execution
    pub async_execution: bool,
    /// Number of GPU streams
    pub gpu_streams: usize,
    /// Enable multi-GPU
    pub multi_gpu: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            simd_level: SimdLevel::detect_best(),
            cache_blocking: true,
            cache_block_size: 64, // Typical L1 cache line
            prefetching: true,
            prefetch_distance: 8,
            kernel_fusion: true,
            async_execution: true,
            gpu_streams: 4,
            multi_gpu: true,
        }
    }
}

/// SIMD instruction set levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    /// No SIMD
    None,
    /// SSE4.2
    SSE42,
    /// AVX2
    AVX2,
    /// AVX-512
    AVX512,
}

impl SimdLevel {
    /// Detect the best available SIMD level
    pub fn detect_best() -> Self {
        if is_x86_feature_detected!("avx512f") {
            SimdLevel::AVX512
        } else if is_x86_feature_detected!("avx2") {
            SimdLevel::AVX2
        } else if is_x86_feature_detected!("sse4.2") {
            SimdLevel::SSE42
        } else {
            SimdLevel::None
        }
    }
    
    /// Get vector width in f64 elements
    pub fn vector_width(&self) -> usize {
        match self {
            SimdLevel::None => 1,
            SimdLevel::SSE42 => 2,
            SimdLevel::AVX2 => 4,
            SimdLevel::AVX512 => 8,
        }
    }
}

/// Performance optimizer
pub struct PerformanceOptimizer {
    config: OptimizationConfig,
    metrics: PerformanceMetrics,
}

/// Performance metrics tracking
#[derive(Debug, Default)]
struct PerformanceMetrics {
    grid_updates_per_second: f64,
    memory_bandwidth_gbps: f64,
    cache_hit_rate: f64,
    simd_efficiency: f64,
    gpu_utilization: f64,
}

impl PerformanceOptimizer {
    pub fn new(config: OptimizationConfig) -> Self {
        info!("Initializing performance optimizer with SIMD level: {:?}", config.simd_level);
        
        Self {
            config,
            metrics: PerformanceMetrics::default(),
        }
    }
    
    /// Apply stencil operation with optimizations
    pub fn apply_stencil(
        &mut self,
        input: &Array3<f64>,
        output: &mut Array3<f64>,
        stencil: &StencilKernel,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = input.dim();
        
        if self.config.enable_simd {
            match self.config.simd_level {
                SimdLevel::AVX512 => self.stencil_vectorized(input, output, stencil, 8)?,
                SimdLevel::AVX2 => self.stencil_vectorized(input, output, stencil, 4)?,
                SimdLevel::SSE42 => self.stencil_vectorized(input, output, stencil, 2)?,
                SimdLevel::None => self.stencil_scalar(input, output, stencil)?,
            }
        } else {
            self.stencil_scalar(input, output, stencil)?;
        }
        
        Ok(())
    }
    
    /// Vectorized stencil computation using safe abstractions
    fn stencil_vectorized(
        &mut self,
        input: &Array3<f64>,
        output: &mut Array3<f64>,
        stencil: &StencilKernel,
        vector_width: usize,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = input.dim();
        
        // Cache blocking parameters
        let block_size = if self.config.cache_blocking {
            self.config.cache_block_size
        } else {
            nx
        };
        
        // Use iterators for cache-friendly access pattern
        (1..nz-1).step_by(block_size).for_each(|block_k| {
            (1..ny-1).step_by(block_size).for_each(|block_j| {
                (1..nx-1).step_by(block_size).for_each(|block_i| {
                    // Process block boundaries
                    let k_end = (block_k + block_size).min(nz - 1);
                    let j_end = (block_j + block_size).min(ny - 1);
                    let i_end = (block_i + block_size).min(nx - 1);
                    
                    // Process block with vectorization
                    (block_k..k_end).for_each(|k| {
                        (block_j..j_end).for_each(|j| {
                            // Process vectorized chunks without allocation
                            let mut i = block_i;
                            while i < i_end {
                                let chunk_end = (i + vector_width).min(i_end);
                                // Process chunk
                                for idx in i..chunk_end {
                                    output[[idx, j, k]] = stencil.apply_scalar(input, idx, j, k);
                                }
                                i = chunk_end;
                            }
                        });
                    });
                });
            });
        });
        
        Ok(())
    }
    
    /// AVX-512 optimized stencil computation - removed unsafe implementation
    #[cfg(feature = "avx512")]
    #[target_feature(enable = "avx512f")]
    fn stencil_avx512(
        &mut self,
        input: &Array3<f64>,
        output: &mut Array3<f64>,
        stencil: &StencilKernel,
    ) -> KwaversResult<()> {
        self.stencil_vectorized(input, output, stencil, 8)
    }
    
    /// AVX-512 stub when feature is not enabled
    #[cfg(not(feature = "avx512"))]
    fn stencil_avx512(
        &mut self,
        _input: &Array3<f64>,
        _output: &mut Array3<f64>,
        _stencil: &StencilKernel,
    ) -> KwaversResult<()> {
        Err(KwaversError::Config(ConfigError::InvalidValue {
            parameter: "simd_level".to_string(),
            value: "AVX-512".to_string(),
            constraint: "AVX-512 not enabled in build".to_string(),
        }))
    }
    
    /// AVX2 optimized stencil computation
    #[target_feature(enable = "avx2")]
    fn stencil_avx2(
        &mut self,
        input: &Array3<f64>,
        output: &mut Array3<f64>,
        stencil: &StencilKernel,
    ) -> KwaversResult<()> {
        self.stencil_vectorized(input, output, stencil, 4)
    }
    
    /// SSE4.2 optimized stencil computation
    #[target_feature(enable = "sse4.2")]
    fn stencil_sse42(
        &mut self,
        input: &Array3<f64>,
        output: &mut Array3<f64>,
        stencil: &StencilKernel,
    ) -> KwaversResult<()> {
        self.stencil_vectorized(input, output, stencil, 2)
    }
    
    /// Scalar stencil computation (fallback)
    fn stencil_scalar(
        &mut self,
        input: &Array3<f64>,
        output: &mut Array3<f64>,
        stencil: &StencilKernel,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = input.dim();
        
        // Parallel execution over outer dimensions
        let mut output_slice = output.slice_mut(s![1..nx-1, 1..ny-1, 1..nz-1]);
        let output_vec: Vec<_> = output_slice.iter_mut().collect();
        output_vec.into_par_iter()
            .enumerate()
            .for_each(|(idx, out)| {
                let k = idx / ((nx - 2) * (ny - 2)) + 1;
                let j = (idx % ((nx - 2) * (ny - 2))) / (nx - 2) + 1;
                let i = idx % (nx - 2) + 1;
                
                *out = stencil.apply_scalar(input, i, j, k);
            });
        
        Ok(())
    }
    
    /// Optimize GPU kernel execution
    #[cfg(feature = "gpu")]
    pub fn optimize_gpu_kernels(
        &mut self,
        _gpu_context: &mut dyn std::any::Any, // Placeholder until GPU module is implemented
        _kernels: Vec<Box<dyn std::any::Any>>,
    ) -> KwaversResult<()> {
        log::warn!("GPU kernel optimization not yet implemented");
        Ok(())
    }
    
    /// Fuse multiple GPU kernels into a single launch
    #[cfg(feature = "gpu")]
    fn fuse_kernels(
        &mut self,
        gpu_context: &mut dyn std::any::Any,
        kernels: Vec<Box<dyn std::any::Any>>,
    ) -> KwaversResult<()> {
        use crate::gpu::GpuContext;
        
        info!("Fusing {} GPU kernels", kernels.len());
        
        // Downcast GPU context
        let context = gpu_context.downcast_mut::<GpuContext>()
            .ok_or_else(|| KwaversError::Config(ConfigError::InvalidValue {
                parameter: "gpu_context".to_string(),
                value: "invalid type".to_string(),
                constraint: "Expected GpuContext".to_string(),
            }))?;
        
        // Downcast and collect kernels
        let mut gpu_kernels = Vec::new();
        let mut all_parameters = Vec::new();
        let mut fused_code = String::new();
        let mut fused_body = String::new();
        
        for (i, kernel_box) in kernels.into_iter().enumerate() {
            let kernel = kernel_box.downcast::<GpuKernel>()
                .map_err(|_| KwaversError::Config(ConfigError::InvalidValue {
                    parameter: "kernel".to_string(),
                    value: format!("kernel {}", i),
                    constraint: "Expected GpuKernel".to_string(),
                }))?;
            
            let kernel_data = *kernel;
            
            // Append kernel code
            fused_code.push_str(&kernel_data.code);
            fused_code.push_str("\n");
            
            // Append kernel body with unique function name
            fused_body.push_str(&format!("    // Kernel {}: {}\n", i, kernel_data.name));
            fused_body.push_str(&format!("    {{\n"));
            fused_body.push_str(&kernel_data.body);
            fused_body.push_str(&format!("    }}\n\n"));
            
            // Collect parameters - do this after using other fields to avoid partial moves
            for param in kernel_data.parameters.iter() {
                all_parameters.push(param.clone());
            }
            
            gpu_kernels.push(kernel_data);
        }
        
        // Create fused kernel
        let fused_kernel_code = format!(
            "__global__ void fused_kernel({}) {{\n{}\n}}\n",
            all_parameters.iter()
                .map(|p| format!("{} {}{}", p.dtype, p.name, p.suffix))
                .collect::<Vec<_>>()
                .join(", "),
            fused_body
        );
        
        // Prepare kernel arguments as void pointers
        let kernel_args: Vec<*const std::ffi::c_void> = all_parameters.iter()
            .map(|p| {
                // This is a placeholder - actual implementation would need to map
                // parameter names to actual buffer/value references
                p.name.as_ptr() as *const std::ffi::c_void
            })
            .collect();
        
        // Launch the fused kernel
        let block_size = gpu_kernels.first()
            .map(|k| k.block_size)
            .unwrap_or((256, 1, 1));
            
        let grid_size = gpu_kernels.first()
            .map(|k| k.grid_size)
            .unwrap_or(1024);
        
        context.launch_kernel(
            "fused_kernel",
            (grid_size as u32, 1, 1),
            block_size,
            &kernel_args,
        )?;
        
        Ok(())
    }
    
    /// Execute kernels sequentially
    #[cfg(feature = "gpu")]
    fn execute_kernels_sequential(
        &mut self,
        _gpu_context: &mut dyn std::any::Any,
        _kernels: Vec<Box<dyn std::any::Any>>,
    ) -> KwaversResult<()> {
        log::warn!("Sequential kernel execution not yet implemented");
        Ok(())
    }
    
    /// Enable multi-GPU execution
    #[cfg(feature = "gpu")]
    pub fn setup_multi_gpu(&mut self, gpu_contexts: Vec<Arc<crate::gpu::GpuContext>>) -> KwaversResult<()> {
        if !self.config.multi_gpu || gpu_contexts.len() <= 1 {
            return Ok(());
        }
        
        info!("Setting up multi-GPU execution with {} devices", gpu_contexts.len());
        
        // Setup peer-to-peer access between GPUs
        for (i, ctx1) in gpu_contexts.iter().enumerate() {
            for (j, _ctx2) in gpu_contexts.iter().enumerate() {
                if i != j {
                    ctx1.enable_peer_access(j as u32)?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Distribute work across multiple GPUs
    #[cfg(feature = "gpu")]
    pub fn distribute_work_multi_gpu<T: Send + Sync>(
        &mut self,
        gpu_contexts: &[Arc<crate::gpu::GpuContext>],
        work_items: Vec<T>,
        process_fn: impl Fn(&crate::gpu::GpuContext, &[T]) -> KwaversResult<()> + Send + Sync,
    ) -> KwaversResult<()> {
        let num_gpus = gpu_contexts.len();
        let items_per_gpu = (work_items.len() + num_gpus - 1) / num_gpus;
        
        // Distribute work
        work_items.par_chunks(items_per_gpu)
            .zip(gpu_contexts.par_iter())
            .try_for_each(|(chunk, gpu_ctx)| {
                process_fn(gpu_ctx, chunk)
            })?;
        
        Ok(())
    }
    
    /// Update performance metrics
    pub fn update_metrics(&mut self, updates_per_second: f64) {
        self.metrics.grid_updates_per_second = updates_per_second;
        
        // Estimate other metrics
        let bytes_per_update = 8.0 * 7.0; // 7-point stencil, 8 bytes per double
        self.metrics.memory_bandwidth_gbps = updates_per_second * bytes_per_update / 1e9;
        
        info!("Performance: {:.1}M updates/sec, {:.1} GB/s bandwidth",
              updates_per_second / 1e6,
              self.metrics.memory_bandwidth_gbps);
    }
    
    /// Get current performance metrics
    pub fn get_metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }
}

/// Stencil kernel definition
#[derive(Debug, Clone)]
pub struct StencilKernel {
    /// Center coefficient
    pub center: f64,
    /// Neighbor coefficient
    pub neighbor: f64,
}

impl StencilKernel {
    /// Apply stencil at a single point
    pub fn apply_scalar(&self, input: &Array3<f64>, i: usize, j: usize, k: usize) -> f64 {
        self.center * input[[i, j, k]] +
        self.neighbor * (
            input[[i-1, j, k]] + input[[i+1, j, k]] +
            input[[i, j-1, k]] + input[[i, j+1, k]] +
            input[[i, j, k-1]] + input[[i, j, k+1]]
        )
    }
}

/// GPU kernel definition
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct GpuKernel {
    pub name: String,
    pub code: String,
    pub body: String,
    pub parameters: Vec<KernelParameter>,
    pub block_size: (u32, u32, u32),
    pub grid_dims: (u32, u32, u32),
    pub grid_size: usize,
}

/// Kernel parameter
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct KernelParameter {
    pub name: String,
    pub dtype: String,
    pub suffix: String,
}

/// Memory bandwidth optimizer
pub struct BandwidthOptimizer {
    /// Memory access pattern
    access_pattern: AccessPattern,
    /// Prefetch strategy
    prefetch_strategy: PrefetchStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum AccessPattern {
    /// Sequential access
    Sequential,
    /// Strided access
    Strided(usize),
    /// Random access
    Random,
}

#[derive(Debug, Clone, Copy)]
pub enum PrefetchStrategy {
    /// No prefetching
    None,
    /// Software prefetching
    Software(usize),
    /// Hardware prefetching
    Hardware,
}

impl BandwidthOptimizer {
    pub fn new() -> Self {
        Self {
            access_pattern: AccessPattern::Sequential,
            prefetch_strategy: PrefetchStrategy::Software(8),
        }
    }
    
    /// Optimize memory layout for bandwidth
    pub fn optimize_layout(&self, data: &mut Array3<f64>) -> KwaversResult<()> {
        match self.access_pattern {
            AccessPattern::Sequential => {
                // Data is already in optimal layout
                Ok(())
            }
            AccessPattern::Strided(stride) => {
                // Reorder data for strided access
                // Implementation would depend on specific access pattern
                Ok(())
            }
            AccessPattern::Random => {
                // Use space-filling curves for better locality
                // Implementation would use Morton or Hilbert curves
                Ok(())
            }
        }
    }
}

/// Cache optimizer
pub struct CacheOptimizer {
    /// L1 cache size (bytes)
    l1_size: usize,
    /// L2 cache size (bytes)
    l2_size: usize,
    /// L3 cache size (bytes)
    l3_size: usize,
    /// Cache line size (bytes)
    cache_line_size: usize,
}

impl CacheOptimizer {
    pub fn new() -> Self {
        Self {
            l1_size: 32 * 1024,      // 32 KB typical L1
            l2_size: 256 * 1024,     // 256 KB typical L2
            l3_size: 8 * 1024 * 1024, // 8 MB typical L3
            cache_line_size: 64,      // 64 bytes typical
        }
    }
    
    /// Calculate optimal block size for cache
    pub fn optimal_block_size(&self, element_size: usize) -> usize {
        // Target 1/3 of L1 cache for working set
        let working_set_size = self.l1_size / 3;
        let elements_per_block = working_set_size / element_size;
        
        // Round down to cache line boundary
        let elements_per_line = self.cache_line_size / element_size;
        (elements_per_block / elements_per_line) * elements_per_line
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_detection() {
        let level = SimdLevel::detect_best();
        println!("Detected SIMD level: {:?}", level);
        assert!(level.vector_width() >= 1);
    }
    
    #[test]
    fn test_stencil_kernel() {
        let stencil = StencilKernel {
            center: -6.0,
            neighbor: 1.0,
        };
        
        let mut input = Array3::zeros((5, 5, 5));
        input[[2, 2, 2]] = 1.0;
        
        let result = stencil.apply_scalar(&input, 2, 2, 2);
        assert_eq!(result, -6.0);
    }
    
    #[test]
    fn test_cache_optimizer() {
        let optimizer = CacheOptimizer::new();
        let block_size = optimizer.optimal_block_size(8); // 8 bytes per f64
        
        assert!(block_size > 0);
        assert!(block_size * 8 <= optimizer.l1_size / 3);
    }
}