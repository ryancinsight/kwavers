/// Operations supported by the GPU accelerator
#[derive(Debug, Clone)]
pub enum GpuOperation {
    /// Acoustic wave propagation (pressure update)
    AcousticPropagation {
        dt: f64,
        sound_speed: f64,
        density: f64,
    },
    /// Electromagnetic wave propagation
    ElectromagneticPropagation {
        dt: f64,
        permittivity: f64,
        permeability: f64,
    },
    /// PDE residual computation for PINN
    PdeResidual,
    /// Matrix operations (FFT, convolution, etc.)
    MatrixOperation,
    /// Custom user-defined operation
    Custom(String),
}

/// Configuration for GPU operations
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// Preferred backend (WGPU, CUDA, etc.)
    pub backend: String,
    /// Memory allocation strategy
    pub memory_strategy: MemoryStrategy,
    /// Precision for computations
    pub precision: Precision,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enable_gpu: true,
            backend: "wgpu".to_string(),
            memory_strategy: MemoryStrategy::Dynamic,
            precision: Precision::F32,
        }
    }
}

/// Memory allocation strategies
#[derive(Debug, Clone)]
pub enum MemoryStrategy {
    /// Pre-allocate fixed memory pools
    Pooled,
    /// Dynamic allocation as needed
    Dynamic,
    /// Memory-mapped for large datasets
    Mapped,
}

/// Precision for GPU computations
#[derive(Debug, Clone, Copy)]
pub enum Precision {
    /// Single precision (f32)
    F32,
    /// Double precision (f64)
    F64,
    /// Mixed precision (f16/f32)
    Mixed,
}

/// Physics parameters for PDE computations
#[derive(Debug, Clone)]
pub struct PhysicsParameters {
    pub equation_type: EquationType,
    pub wave_speed: Option<f64>,
    pub diffusion_coefficient: Option<f64>,
    pub thermal_conductivity: Option<f64>,
    pub viscosity: Option<f64>,
    pub dt: f64,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

/// Types of PDE equations supported
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EquationType {
    Wave,
    Heat,
    Diffusion,
    NavierStokes,
}
