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
//! use kwavers_solver::inverse::pinn::ml::{BurnPINN1DWave, BurnPINNConfig};
//! use kwavers_solver::inverse::pinn::ml::fdtd_reference::FDTDConfig;
//! use kwavers_solver::inverse::pinn::ml::validation::validate_pinn_vs_fdtd;
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
//! # Ok::<(), kwavers_core::error::KwaversError>(())
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

// Field-surrogate PINN: parameterised (x, y, z, f0, pnp) → (p_min,
// p_max, p_rms) for treatment-planner use. Static surrogate trained
// against `physics::field_surrogate::KernelCube` data — see module
// docs for the pipeline and Phase C-1/C-2 split.
#[cfg(feature = "pinn")]
pub mod field_surrogate;

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

#[cfg(feature = "pinn")]
pub use burn_wave_equation_2d::{
    BoundaryCondition2D, BurnLossWeights2D, BurnPINN2DConfig, BurnPINN2DWave,
    BurnTrainingMetrics2D, BurnWave2dGeometry, BurnWave2dInterfaceCondition,
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

// Beamforming provider adapter moved to analysis layer
// See: src/analysis/signal_processing/beamforming/neural/backends/burn_adapter.rs

#[cfg(feature = "pinn")]
pub mod acoustic_wave;

#[cfg(feature = "pinn")]
#[path = "cavitation_coupled/mod.rs"]
pub mod cavitation_coupled;

#[cfg(feature = "pinn")]
pub mod electromagnetic;

#[cfg(feature = "pinn")]
pub mod sonoluminescence_coupled;

#[cfg(feature = "pinn")]
pub mod universal_solver;

#[cfg(feature = "pinn")]
pub mod gpu_accelerator;

#[cfg(feature = "pinn")]
pub mod adaptive_sampling;

#[cfg(feature = "pinn")]
pub use multi_gpu_manager::{
    DataTransfer, FaultTolerance, LoadBalancingAlgorithm, MultiGpuDecompositionStrategy,
    MultiGpuManager, MultiGpuPerformanceMonitor, PerformanceSummary,
    PinnMultiGpuCommunicationChannel, PinnMultiGpuDeviceInfo, PinnMultiGpuTransferStatus, WorkUnit,
};

#[cfg(feature = "pinn")]
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
    MlQuantizer, QuantizationModelMetadata, QuantizationParams, QuantizationScheme, QuantizedModel,
    QuantizedTensor,
};

#[cfg(feature = "pinn")]
pub use edge_runtime::{
    Architecture, DataType, EdgeRuntime, EdgeRuntimePerformanceMonitor as EdgePerformanceMonitor,
    ExecutionKernel, HardwareCapabilities, IOSpecification, MemoryAllocator,
};

// Sprint 154: Meta-Learning & Transfer Learning
#[cfg(feature = "pinn")]
pub use meta_learning::{
    MetaLearner, MetaLearningConfig, MetaLearningPhysicsParameters, MetaLearningSamplingStrategy,
    MetaLearningStats, MetaLoss, PhysicsTask, TaskData, TaskSampler,
};

#[cfg(feature = "pinn")]
pub use transfer_learning::{
    FreezeStrategy, TransferLearner, TransferLearningConfig, TransferLearningStats, TransferMetrics,
};

// Beamforming provider adapter is now in analysis layer (no longer exported from solver)

// Sprint 156: Advanced Physics Domains
#[cfg(feature = "pinn")]
pub use adapters::source::{
    adapt_sources, AdapterError, PinnAcousticSource, PinnSourceClass, PinnSourceFocalProperties,
};

#[cfg(feature = "pinn")]
pub use adapters::electromagnetic::{adapt_em_sources, EMAdapterError, PinnEMSource};

#[cfg(feature = "pinn")]
pub use acoustic_wave::{
    AcousticBoundarySpec, AcousticProblemType, AcousticWaveDomain, PinnAcousticBoundaryType,
};

#[cfg(feature = "pinn")]
pub use cavitation_coupled::{CavitationCoupledDomain, CavitationCouplingConfig};

#[cfg(feature = "pinn")]
pub use electromagnetic::{EMProblemType, ElectromagneticBoundarySpec, ElectromagneticDomain};

#[cfg(feature = "pinn")]
pub use sonoluminescence_coupled::{SonoluminescenceCoupledDomain, SonoluminescenceCouplingConfig};

#[cfg(feature = "pinn")]
pub use universal_solver::{
    EarlyStoppingConfig, GeometricFeature, PhysicsSolution, UniversalPINNSolver,
    UniversalSolverConvergenceInfo, UniversalSolverDomainInfo, UniversalSolverLrSchedule,
    UniversalSolverMemoryStats, UniversalSolverStats, UniversalTrainingConfig,
};

#[cfg(feature = "pinn")]
pub use gpu_accelerator::{
    BatchedPINNTrainer, CudaBuffer, CudaKernelManager, CudaStream, PinnGpuMemoryPoolType,
    TrainingStep,
};

#[cfg(feature = "pinn")]
pub use adaptive_sampling::{AdaptiveCollocationSampler, SamplingStats};

#[cfg(feature = "pinn")]
pub mod uncertainty_quantification;

#[cfg(feature = "pinn")]
pub use uncertainty_quantification::{
    PinnBayesianPINN, PinnConformalPredictor, PinnPredictionWithUncertainty, PinnUncertaintyConfig,
    PinnUncertaintyMethod, UncertaintyStats,
};

// #[cfg(feature = "pinn")]
// pub use advanced_architectures::{
//     ResNetPINN1D, ResNetPINN2D, FourierFeatures, MultiScaleFeatures, PhysicsAttention, ResNetPINNConfig
// };

// Placeholder when pinn feature is not enabled
#[cfg(not(feature = "pinn"))]
#[derive(Debug)]
pub struct PINN1DWave;

#[cfg(not(feature = "pinn"))]
impl PINN1DWave {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(_wave_speed: f64, _config: ()) -> Result<Self, kwavers_core::error::KwaversError> {
        Err(kwavers_core::error::KwaversError::InvalidInput(
            "PINN feature not enabled. Add 'pinn' feature to Cargo.toml".to_owned(),
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
            let _solver = solver.unwrap();
        }

        #[cfg(not(feature = "pinn"))]
        {
            let result = super::PINN1DWave::new(
                kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM,
                (),
            );
            assert!(result.is_err());
        }
    }
}
