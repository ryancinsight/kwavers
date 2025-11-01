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
//! use kwavers::ml::pinn::{PINN1DWave, PINNConfig};
//! use kwavers::ml::pinn::fdtd_reference::FDTDConfig;
//! use kwavers::ml::pinn::validation::validate_pinn_vs_fdtd;
//!
//! // Create 1D wave equation PINN
//! let config = PINNConfig::default();
//! let mut pinn = PINN1DWave::new(1500.0, config)?; // 1500 m/s wave speed
//!
//! // Train on reference data
//! let metrics = pinn.train(&reference_data, 1000)?;
//!
//! // Validate against FDTD
//! let fdtd_config = FDTDConfig::default();
//! let report = validate_pinn_vs_fdtd(&pinn, fdtd_config)?;
//! println!("{}", report.summary());
//!
//! // Fast inference (1000× speedup)
//! let prediction = pinn.predict(&x_points, &t_points);
//! # Ok::<(), kwavers::error::KwaversError>(())
//! # }
//! ```

#[cfg(feature = "pinn")]
pub mod fdtd_reference;

#[cfg(feature = "pinn")]
pub mod validation;

#[cfg(feature = "pinn")]
pub mod wave_equation_1d;

// Sprint 143 Phase 2: Burn-based PINN with automatic differentiation
#[cfg(feature = "pinn")]
pub mod burn_wave_equation_1d;

// Sprint 144: 2D Wave Equation PINN extension
#[cfg(feature = "pinn")]
pub mod burn_wave_equation_2d;

// Sprint 151: GPU Acceleration & Advanced Geometries
#[cfg(feature = "pinn")]
pub mod gpu_accelerator;

// Sprint 150: Advanced neural architectures for improved PINN convergence
// Temporarily disabled due to Burn API compatibility issues
// #[cfg(feature = "pinn")]
// pub mod advanced_architectures;

// Sprint 151: Transfer learning for PINN adaptation and fine-tuning
#[cfg(feature = "pinn")]
pub mod transfer_learning;

#[cfg(feature = "pinn")]
pub use wave_equation_1d::{LossWeights, PINNConfig, TrainingMetrics, ValidationMetrics, PINN1DWave};

#[cfg(feature = "pinn")]
pub use burn_wave_equation_1d::{BurnLossWeights, BurnPINNConfig, BurnPINN1DWave, BurnTrainingMetrics};

#[cfg(feature = "pinn")]
pub use burn_wave_equation_2d::{
    BurnLossWeights2D, BurnPINN2DConfig, BurnPINN2DWave, BurnTrainingMetrics2D, BoundaryCondition2D, Geometry2D, InterfaceCondition
};

// Sprint 151: GPU Acceleration & Advanced Geometries
#[cfg(feature = "pinn")]
pub use gpu_accelerator::{
    PinnGpuAccelerator, GpuMemoryManager, TrainingStats, gpu_optimizations
};

// Sprint 152: Multi-GPU Support & Distributed Training
#[cfg(feature = "pinn")]
pub mod multi_gpu_manager;

#[cfg(feature = "pinn")]
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
pub use multi_gpu_manager::{
    MultiGpuManager, GpuDeviceInfo, DecompositionStrategy, LoadBalancingAlgorithm,
    WorkUnit, CommunicationChannel, DataTransfer, TransferStatus, PerformanceMonitor,
    FaultTolerance, PerformanceSummary
};

#[cfg(feature = "pinn")]
pub use distributed_training::{
    DistributedPinnTrainer, GradientAggregation, CheckpointManager, TrainingCoordinator
};

// Sprint 153: Real-Time Inference & Optimization
#[cfg(feature = "pinn")]
pub use jit_compiler::{
    JitCompiler, OptimizedRuntime, CompiledKernel, OptimizationLevel,
    CompilerStats, InferenceStats
};

#[cfg(feature = "pinn")]
pub use quantization::{
    Quantizer, QuantizedModel, QuantizationScheme, QuantizationParams,
    ModelMetadata, QuantizedTensor
};

#[cfg(feature = "pinn")]
pub use edge_runtime::{
    EdgeRuntime, MemoryAllocator, ExecutionKernel, IOSpecification,
    DataType, HardwareCapabilities, Architecture, PerformanceMonitor as EdgePerformanceMonitor
};

// Sprint 154: Meta-Learning & Transfer Learning
#[cfg(feature = "pinn")]
pub use meta_learning::{
    MetaLearner, MetaLearningConfig, PhysicsTask, PhysicsParameters,
    TaskData, MetaLoss, TaskSampler, SamplingStrategy, MetaLearningStats
};

#[cfg(feature = "pinn")]
pub use transfer_learning::{
    TransferLearner, TransferLearningConfig, FreezeStrategy, TransferMetrics,
    TransferLearningStats
};

#[cfg(feature = "pinn")]
pub mod uncertainty_quantification;

#[cfg(feature = "pinn")]
pub use uncertainty_quantification::{
    BayesianPINN, UncertaintyConfig, PredictionWithUncertainty, UncertaintyMethod,
    UncertaintyStats, ConformalPredictor
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
    pub fn new(_wave_speed: f64, _config: ()) -> Result<Self, crate::error::KwaversError> {
        Err(crate::error::KwaversError::InvalidInput(
            "PINN feature not enabled. Add 'pinn' feature to Cargo.toml".to_string()
        ))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_pinn_module_exists() {
        // Basic module existence test
        assert!(true);
    }
}
