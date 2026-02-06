//! GPU Physics Kernel Registry and Dispatcher
//!
//! Manages compilation, configuration, and dispatch of GPU compute kernels
//! for multiphysics simulations (acoustic, elastic, optical, thermal).

use crate::core::error::KwaversResult;
use std::collections::HashMap;

/// Physics domain for GPU kernel dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysicsDomain {
    /// Acoustic FDTD time-domain propagation
    AcousticFDTD,

    /// Elastic wave stress-velocity coupling
    ElasticWaves,

    /// K-space spectral methods
    KSpaceMethods,

    /// Absorption and damping (PML)
    Absorption,

    /// Field interpolation for multi-GPU (Phase 1)
    FieldInterpolation,
}

impl PhysicsDomain {
    /// Get domain name for identification
    pub fn name(&self) -> &'static str {
        match self {
            Self::AcousticFDTD => "acoustic_fdtd",
            Self::ElasticWaves => "elastic_waves",
            Self::KSpaceMethods => "kspace_methods",
            Self::Absorption => "absorption",
            Self::FieldInterpolation => "field_interpolation",
        }
    }
}

/// Workgroup configuration for GPU kernels
#[derive(Debug, Clone)]
pub struct WorkgroupConfig {
    /// X dimension workgroup size
    pub workgroup_x: u32,

    /// Y dimension workgroup size
    pub workgroup_y: u32,

    /// Z dimension workgroup size
    pub workgroup_z: u32,

    /// Total threads per workgroup
    pub total_threads: u32,
}

impl WorkgroupConfig {
    /// Create workgroup config for problem size
    pub fn new(nx: u32, ny: u32, nz: u32) -> Self {
        // Optimal for most GPUs: 8×8×8 = 512 threads
        let config = match (nx, ny, nz) {
            // Small problems: use smaller workgroups
            (x, y, z) if x <= 64 && y <= 64 && z <= 64 => Self {
                workgroup_x: 4,
                workgroup_y: 4,
                workgroup_z: 4,
            },
            // Medium problems: balanced workgroup
            (x, y, z) if x <= 256 && y <= 256 && z <= 256 => Self {
                workgroup_x: 8,
                workgroup_y: 8,
                workgroup_z: 8,
            },
            // Large problems: larger workgroups
            _ => Self {
                workgroup_x: 16,
                workgroup_y: 8,
                workgroup_z: 4,
            },
        };

        Self {
            total_threads: config.workgroup_x * config.workgroup_y * config.workgroup_z,
            ..config
        }
    }
}

/// Physics kernel information
#[derive(Debug, Clone)]
pub struct PhysicsKernel {
    /// Domain this kernel handles
    pub domain: PhysicsDomain,

    /// WGSL shader source code
    pub shader_source: String,

    /// Compute shader entry point name
    pub entry_point: String,

    /// Estimated FLOPs per grid element
    pub flops_per_element: u64,

    /// Optimal workgroup size
    pub workgroup_config: WorkgroupConfig,
}

impl PhysicsKernel {
    /// Create new physics kernel
    pub fn new(
        domain: PhysicsDomain,
        shader_source: String,
        entry_point: String,
        flops_per_element: u64,
        workgroup_config: WorkgroupConfig,
    ) -> Self {
        Self {
            domain,
            shader_source,
            entry_point,
            flops_per_element,
            workgroup_config,
        }
    }

    /// Estimate kernel execution time on GPU
    ///
    /// Uses: time = (grid_elements × flops_per_element) / gpu_bandwidth
    /// Typical GPU: 10 TFLOP/s = 10e12 FLOP/s
    pub fn estimate_time_ms(&self, num_elements: usize) -> f64 {
        let total_flops = (num_elements as u64) * self.flops_per_element;
        let gpu_flops_per_sec = 10e12; // Typical modern GPU
        (total_flops as f64 / gpu_flops_per_sec) * 1000.0
    }
}

/// Registry of physics kernels
#[derive(Debug)]
pub struct PhysicsKernelRegistry {
    /// Registered kernels by domain
    kernels: HashMap<PhysicsDomain, PhysicsKernel>,

    /// Workgroup configurations cache
    workgroup_cache: HashMap<String, WorkgroupConfig>,
}

impl PhysicsKernelRegistry {
    /// Create new kernel registry
    pub fn new() -> Self {
        Self {
            kernels: HashMap::new(),
            workgroup_cache: HashMap::new(),
        }
    }

    /// Register a physics kernel
    pub fn register(&mut self, kernel: PhysicsKernel) -> KwaversResult<()> {
        self.kernels.insert(kernel.domain, kernel);
        Ok(())
    }

    /// Get registered kernel for domain
    pub fn get_kernel(&self, domain: PhysicsDomain) -> Option<&PhysicsKernel> {
        self.kernels.get(&domain)
    }

    /// Get optimal workgroup config for grid size
    pub fn get_workgroup_config(
        &mut self,
        domain: PhysicsDomain,
        nx: u32,
        ny: u32,
        nz: u32,
    ) -> WorkgroupConfig {
        let cache_key = format!("{}_{}x{}x{}", domain.name(), nx, ny, nz);

        if !self.workgroup_cache.contains_key(&cache_key) {
            let config = WorkgroupConfig::new(nx, ny, nz);
            self.workgroup_cache
                .insert(cache_key.clone(), config.clone());
            config
        } else {
            self.workgroup_cache[&cache_key].clone()
        }
    }

    /// Estimate total execution time for all kernels
    pub fn estimate_total_time_ms(&self, num_elements: usize) -> f64 {
        self.kernels
            .values()
            .map(|k| k.estimate_time_ms(num_elements))
            .sum()
    }

    /// List all registered kernels
    pub fn list_kernels(&self) -> Vec<PhysicsDomain> {
        self.kernels.keys().copied().collect()
    }

    /// Check if all required domains are registered
    pub fn is_complete(&self, required_domains: &[PhysicsDomain]) -> bool {
        required_domains
            .iter()
            .all(|d| self.kernels.contains_key(d))
    }
}

impl Default for PhysicsKernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physics_domain_names() {
        assert_eq!(PhysicsDomain::AcousticFDTD.name(), "acoustic_fdtd");
        assert_eq!(PhysicsDomain::Absorption.name(), "absorption");
    }

    #[test]
    fn test_workgroup_config_small_grid() {
        let config = WorkgroupConfig::new(32, 32, 32);
        assert_eq!(config.workgroup_x, 4);
        assert_eq!(config.total_threads, 64);
    }

    #[test]
    fn test_workgroup_config_medium_grid() {
        let config = WorkgroupConfig::new(128, 128, 128);
        assert_eq!(config.workgroup_x, 8);
        assert_eq!(config.total_threads, 512);
    }

    #[test]
    fn test_workgroup_config_large_grid() {
        let config = WorkgroupConfig::new(512, 512, 512);
        assert_eq!(config.workgroup_x, 16);
        assert_eq!(config.total_threads, 512);
    }

    #[test]
    fn test_kernel_creation() {
        let kernel = PhysicsKernel::new(
            PhysicsDomain::AcousticFDTD,
            "fn main() {}".to_string(),
            "compute_main".to_string(),
            25,
            WorkgroupConfig::new(64, 64, 64),
        );

        assert_eq!(kernel.domain, PhysicsDomain::AcousticFDTD);
        assert_eq!(kernel.flops_per_element, 25);
    }

    #[test]
    fn test_kernel_time_estimation() {
        let kernel = PhysicsKernel::new(
            PhysicsDomain::AcousticFDTD,
            "fn main() {}".to_string(),
            "compute_main".to_string(),
            25,
            WorkgroupConfig::new(64, 64, 64),
        );

        let time_ms = kernel.estimate_time_ms(1_000_000);
        // 1M elements × 25 FLOP/element / 10e12 FLOP/s × 1000 = 0.0025 ms
        assert!(time_ms < 1.0); // Should be very fast
    }

    #[test]
    fn test_registry_operations() {
        let mut registry = PhysicsKernelRegistry::new();

        let kernel = PhysicsKernel::new(
            PhysicsDomain::AcousticFDTD,
            "fn main() {}".to_string(),
            "compute_main".to_string(),
            25,
            WorkgroupConfig::new(64, 64, 64),
        );

        registry.register(kernel).unwrap();

        assert!(registry.get_kernel(PhysicsDomain::AcousticFDTD).is_some());
        assert!(registry.get_kernel(PhysicsDomain::Absorption).is_none());
    }

    #[test]
    fn test_workgroup_caching() {
        let mut registry = PhysicsKernelRegistry::new();

        let config1 = registry.get_workgroup_config(PhysicsDomain::AcousticFDTD, 128, 128, 128);
        let config2 = registry.get_workgroup_config(PhysicsDomain::AcousticFDTD, 128, 128, 128);

        assert_eq!(config1.workgroup_x, config2.workgroup_x);
        assert_eq!(registry.workgroup_cache.len(), 1);
    }

    #[test]
    fn test_completeness_check() {
        let mut registry = PhysicsKernelRegistry::new();

        let kernel = PhysicsKernel::new(
            PhysicsDomain::AcousticFDTD,
            "fn main() {}".to_string(),
            "compute_main".to_string(),
            25,
            WorkgroupConfig::new(64, 64, 64),
        );

        registry.register(kernel).unwrap();

        let required = vec![PhysicsDomain::AcousticFDTD];
        assert!(registry.is_complete(&required));

        let required_more = vec![PhysicsDomain::AcousticFDTD, PhysicsDomain::Absorption];
        assert!(!registry.is_complete(&required_more));
    }
}
