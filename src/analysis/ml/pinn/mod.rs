//! Physics-Informed Neural Networks (PINNs) for Ultrasound Simulation
//!
//! This module implements PINNs for solving wave equations with 1000× faster inference
//! compared to traditional FDTD methods. PINNs embed physical laws (PDEs) directly
//! into the neural network loss function, ensuring physics consistency.
//!
//! ## Overview
//!
//! Physics-Informed Neural Networks solve partial differential equations by:
//! 1. Training a neural network to approximate the solution u(x,t)
//! 2. Enforcing PDE residuals through automatic differentiation
//! 3. Incorporating boundary and initial conditions in the loss
//!
//! ## Architecture
//!
//! ```text
//! Input: (x, t) → Neural Network → Output: u(x, t)
//!                      ↓
//!          Automatic Differentiation
//!                      ↓
//!        PDE Residual: ∂²u/∂t² - c²∂²u/∂x²
//! ```
//!
//! ## Sprint 143 Enhancements
//!
//! - FDTD reference solution generator for validation
//! - Comprehensive validation framework comparing PINN vs FDTD
//! - Burn 0.18 integration (bincode compatibility resolved)
//! - Performance benchmarking with speedup measurements
//!
//! ## Literature References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks: A deep learning framework"
//!   *Journal of Computational Physics*, 378, 686-707.
//! - Raissi et al. (2017): "Hidden physics models: Machine learning of nonlinear PDEs"
//!   *Journal of Computational Physics*, 357, 125-141.
//!
//! ## Example
//!
//! ```no_run
//! # #[cfg(feature = "pinn")]
//! # {
//! use kwavers::ml::pinn::{BurnPINN1DWave, BurnPINNConfig};
//! use kwavers::ml::pinn::fdtd_reference::FDTDConfig;
//! use kwavers::ml::pinn::validation::validate_pinn_vs_fdtd;
//!
//! // Create 1D wave equation PINN
//! let device = Default::default();
//! let config = BurnPINNConfig::default();
//! let mut pinn = BurnPINN1DWave::new(config, &device)?; // 1500 m/s wave speed
//!
//! // Train on reference data
//! let metrics = pinn.train(&x_points, &t_points, &reference_data, 1500.0, &device, 1000)?;
//!
//! // Validate against FDTD
//! let fdtd_config = FDTDConfig::default();
//! let report = validate_pinn_vs_fdtd(&pinn, fdtd_config)?;
//! println!("{}", report.summary());
//!
//! // Fast inference (1000× speedup)
//! let prediction = pinn.predict(&x_points, &t_points, &device)?;
//! # Ok::<(), kwavers::error::KwaversError>(())
//! # }
//! ```

#[cfg(feature = "pinn")]
pub mod adapters;

#[cfg(feature = "pinn")]
pub mod fdtd_reference;

#[cfg(feature = "pinn")]
pub mod validation;

// Sprint 143 Phase 2: Burn-based PINN with automatic differentiation
#[cfg(feature = "pinn")]
pub mod burn_wave_equation_1d;

// Sprint 144: 2D Wave Equation PINN extension
#[cfg(feature = "pinn")]
pub mod burn_wave_equation_2d;

// Sprint 173: 3D Wave Equation PINN extension for heterogeneous media
#[cfg(feature = "pinn")]
pub mod burn_wave_equation_3d;

// Sprint 151: GPU Acceleration & Advanced Geometries
// #[cfg(feature = "pinn")]
// pub mod gpu_accelerator;

// Sprint 150: Advanced neural architectures for improved PINN convergence
#[cfg(feature = "pinn")]
pub mod advanced_architectures;

// Sprint 191: Burn autodiff utilities for gradient computation patterns
#[cfg(feature = "pinn")]
pub mod autodiff_utils;

// Sprint 151: Transfer learning for PINN adaptation and fine-tuning
#[cfg(feature = "pinn")]
pub mod transfer_learning;

#[cfg(feature = "pinn")]
pub use burn_wave_equation_1d::{
    BurnLossWeights, BurnPINN1DWave, BurnPINNConfig, BurnPINNTrainer, BurnTrainingMetrics,
    SimpleOptimizer,
};

#[cfg(all(feature = "pinn", feature = "api"))]
pub mod trainer;

#[cfg(all(feature = "pinn", feature = "api"))]
pub use trainer::{Geometry, PINNConfig, PINNTrainer, PhysicsParams, TrainingConfig};

#[cfg(feature = "pinn")]
pub use burn_wave_equation_2d::{
    BoundaryCondition2D, BurnLossWeights2D, BurnPINN2DConfig, BurnPINN2DWave,
    BurnTrainingMetrics2D, Geometry2D, InterfaceCondition,
};

#[cfg(feature = "pinn")]
pub use burn_wave_equation_3d::{
    BoundaryCondition3D, BurnLossWeights3D, BurnPINN3DConfig, BurnPINN3DWave,
    BurnTrainingMetrics3D, Geometry3D, InterfaceCondition3D,
};

// Sprint 151: GPU Acceleration & Advanced Geometries
#[cfg(feature = "pinn")]
pub use gpu_accelerator::{GpuMemoryManager, TrainingStats};

// Sprint 152: Multi-GPU Support & Distributed Training
#[cfg(feature = "pinn")]
pub mod multi_gpu_manager;

#[cfg(all(feature = "pinn", feature = "api"))]
pub mod distributed_training;

// Sprint 153: Real-Time Inference & Optimization
#[cfg(feature = "pinn")]
pub mod jit_compiler;

#[cfg(feature = "pinn")]
pub mod quantization;

#[cfg(feature = "pinn")]
pub mod edge_runtime;

// Sprint 154: Meta-Learning & Transfer Learning
#[cfg(feature = "pinn")]
pub mod meta_learning;

// Sprint 156: Advanced Physics Domains for PINN
#[cfg(feature = "pinn")]
pub mod physics;

#[cfg(feature = "pinn")]
pub mod acoustic_wave;

#[cfg(feature = "pinn")]
pub mod cavitation_coupled;

#[cfg(feature = "pinn")]
pub mod electromagnetic;

#[cfg(feature = "pinn")]
pub mod sonoluminescence_coupled;

#[cfg(all(feature = "pinn", feature = "gpu"))]
pub mod electromagnetic_gpu;

#[cfg(feature = "pinn")]
pub mod universal_solver;

#[cfg(feature = "pinn")]
pub mod gpu_accelerator;

#[cfg(feature = "pinn")]
pub mod adaptive_sampling;

#[cfg(feature = "pinn")]
pub use multi_gpu_manager::{
    CommunicationChannel, DataTransfer, DecompositionStrategy, FaultTolerance, GpuDeviceInfo,
    LoadBalancingAlgorithm, MultiGpuManager, PerformanceMonitor, PerformanceSummary,
    TransferStatus, WorkUnit,
};

#[cfg(all(feature = "pinn", feature = "api"))]
pub use distributed_training::{
    CheckpointManager, DistributedPinnTrainer, GradientAggregation, TrainingCoordinator,
};

// Sprint 153: Real-Time Inference & Optimization
#[cfg(feature = "pinn")]
pub use jit_compiler::{
    CompiledKernel, CompilerStats, InferenceStats, JitCompiler, OptimizationLevel, OptimizedRuntime,
};

#[cfg(feature = "pinn")]
pub use quantization::{
    ModelMetadata, QuantizationParams, QuantizationScheme, QuantizedModel, QuantizedTensor,
    Quantizer,
};

#[cfg(feature = "pinn")]
pub use edge_runtime::{
    Architecture, DataType, EdgeRuntime, ExecutionKernel, HardwareCapabilities, IOSpecification,
    MemoryAllocator, PerformanceMonitor as EdgePerformanceMonitor,
};

// Sprint 154: Meta-Learning & Transfer Learning
#[cfg(feature = "pinn")]
pub use meta_learning::{
    MetaLearner, MetaLearningConfig, MetaLearningStats, MetaLoss, PhysicsParameters, PhysicsTask,
    SamplingStrategy, TaskData, TaskSampler,
};

#[cfg(feature = "pinn")]
pub use transfer_learning::{
    FreezeStrategy, TransferLearner, TransferLearningConfig, TransferLearningStats, TransferMetrics,
};

// Sprint 156: Advanced Physics Domains
#[cfg(feature = "pinn")]
pub use adapters::source::{
    adapt_sources, AdapterError, FocalProperties, PinnAcousticSource, PinnSourceClass,
};

#[cfg(feature = "pinn")]
pub use adapters::electromagnetic::{adapt_em_sources, EMAdapterError, PinnEMSource};

#[cfg(feature = "pinn")]
pub use acoustic_wave::{
    AcousticBoundarySpec, AcousticBoundaryType, AcousticProblemType, AcousticWaveDomain,
};

#[cfg(feature = "pinn")]
pub use cavitation_coupled::{CavitationCoupledDomain, CavitationCouplingConfig};

#[cfg(feature = "pinn")]
pub use electromagnetic::{EMProblemType, ElectromagneticBoundarySpec, ElectromagneticDomain};

#[cfg(feature = "pinn")]
pub use sonoluminescence_coupled::{SonoluminescenceCoupledDomain, SonoluminescenceCouplingConfig};

#[cfg(all(feature = "pinn", feature = "gpu"))]
pub use electromagnetic_gpu::{BoundaryCondition, EMConfig, EMFieldData, GPUEMSolver};

#[cfg(feature = "pinn")]
pub use universal_solver::{
    ConvergenceInfo, DomainInfo, EarlyStoppingConfig, GeometricFeature, LearningRateSchedule,
    MemoryStats, PhysicsSolution, UniversalPINNSolver, UniversalSolverStats,
    UniversalTrainingConfig,
};

#[cfg(feature = "pinn")]
pub use gpu_accelerator::{
    BatchedPINNTrainer, CudaBuffer, CudaKernelManager, CudaStream, MemoryPoolType, TrainingStep,
};

#[cfg(feature = "pinn")]
pub use adaptive_sampling::{AdaptiveCollocationSampler, SamplingStats};

#[cfg(feature = "pinn")]
pub mod uncertainty_quantification;

#[cfg(feature = "pinn")]
pub use uncertainty_quantification::{
    BayesianPINN, ConformalPredictor, PinnUncertaintyConfig, PredictionWithUncertainty,
    UncertaintyMethod, UncertaintyStats,
};

// #[cfg(feature = "pinn")]
// pub use advanced_architectures::{
//     ResNetPINN1D, ResNetPINN2D, FourierFeatures, MultiScaleFeatures, PhysicsAttention, ResNetPINNConfig
// };

// Placeholder when pinn feature is not enabled
#[cfg(not(feature = "pinn"))]
pub struct PINN1DWave;

#[cfg(not(feature = "pinn"))]
impl PINN1DWave {
    pub fn new(_wave_speed: f64, _config: ()) -> Result<Self, crate::core::error::KwaversError> {
        Err(crate::core::error::KwaversError::InvalidInput(
            "PINN feature not enabled. Add 'pinn' feature to Cargo.toml".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_pinn_module_exists() {
        #[cfg(feature = "pinn")]
        {
            use burn::backend::{Autodiff, NdArray};
            type TestBackend = Autodiff<NdArray<f32>>;
            let solver = super::UniversalPINNSolver::<TestBackend>::new();
            assert!(solver.is_ok());
        }

        #[cfg(not(feature = "pinn"))]
        {
            let result = super::PINN1DWave::new(1500.0, ());
            assert!(result.is_err());
        }
    }
}
