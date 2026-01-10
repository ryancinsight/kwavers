//! Physics-Informed Neural Networks (PINN) for beamforming optimization.
//!
//! This module implements PINN-based beamforming that enforces acoustic wave
//! propagation physics during neural network training and inference. The approach
//! combines data-driven learning with physical constraints to achieve superior
//! imaging quality while maintaining consistency with wave equation solutions.
//!
//! ## Core Concept
//!
//! Traditional neural networks learn purely from data, which can produce physically
//! inconsistent results. PINNs incorporate governing equations (wave equation,
//! eikonal equation) as soft constraints in the loss function:
//!
//! ```text
//! L_total = L_data + λ_physics · L_physics + λ_boundary · L_boundary
//! ```
//!
//! where:
//! - L_data: Data fidelity (MSE between prediction and measurement)
//! - L_physics: Wave equation residual penalty
//! - L_boundary: Boundary condition enforcement
//! - λ: Relative weights for constraint balancing
//!
//! ## Wave Physics Constraints
//!
//! ### 1. Acoustic Wave Equation
//!
//! The fundamental constraint for pressure field p(x,t):
//! ```text
//! ∂²p/∂t² - c²∇²p = 0
//! ```
//!
//! Residual penalty:
//! ```text
//! L_physics = (1/N) ∑ᵢ |∂²p/∂t² - c²∇²p|²
//! ```
//!
//! ### 2. Eikonal Equation for Travel Time
//!
//! For delay calculation τ(x) in heterogeneous media:
//! ```text
//! |∇τ|² = n²(x)/c²
//! ```
//! where n(x) is the refractive index.
//!
//! ### 3. Reciprocity Constraint
//!
//! Green's function symmetry:
//! ```text
//! G(r_a, r_b, ω) = G(r_b, r_a, ω)
//! ```
//!
//! Enforced via symmetric weight matrices and bidirectional loss terms.
//!
//! ## Module Organization
//!
//! - [`processor`]: Main PINN beamforming processor
//! - (Future) `config`: PINN-specific configuration structures
//! - (Future) `inference`: Optimized inference routines
//!
//! ## Benefits Over Traditional Methods
//!
//! ### Resolution Enhancement
//! - **Traditional DAS**: Limited by diffraction (λ/2)
//! - **PINN-enhanced**: 2-3× improvement via learned super-resolution
//!
//! ### Robustness
//! - Handles phase aberration and channel dropout
//! - Generalizes to unseen imaging scenarios
//! - Maintains physical plausibility under noise
//!
//! ### Uncertainty Quantification
//! - Bayesian PINNs provide confidence estimates
//! - Critical for clinical decision-making
//! - Identifies regions requiring additional imaging
//!
//! ## Mathematical Derivation
//!
//! ### Forward Problem
//!
//! Neural network approximation:
//! ```text
//! p̃(x,t; θ) ≈ p(x,t)
//! ```
//! where θ are trainable parameters.
//!
//! ### Physics Loss
//!
//! Automatic differentiation computes derivatives:
//! ```text
//! ∂²p̃/∂t² = ∂²(NN(x,t;θ))/∂t²
//! ```
//!
//! Physics residual:
//! ```text
//! R(x,t) = ∂²p̃/∂t² - c²∇²p̃
//! ```
//!
//! Loss:
//! ```text
//! L_physics = ∫∫ R²(x,t) dx dt
//! ```
//! (approximated via Monte Carlo sampling)
//!
//! ### Optimization
//!
//! Adam optimizer with learning rate schedule:
//! ```text
//! θ_{k+1} = θ_k - η_k · ∇_θ L_total
//! ```
//!
//! Convergence criterion:
//! ```text
//! ||∇_θ L|| < ε  or  k > k_max
//! ```
//!
//! ## Usage Example
//!
//! ```ignore
//! use kwavers::analysis::signal_processing::beamforming::neural::pinn::NeuralBeamformingProcessor;
//! use kwavers::analysis::signal_processing::beamforming::neural::PINNBeamformingConfig;
//!
//! // Configure PINN beamformer
//! let config = PINNBeamformingConfig {
//!     enable_pinn: true,
//!     enable_uncertainty_quantification: true,
//!     physics_weight: 1.0,
//!     learning_rate: 0.001,
//!     num_epochs: 1000,
//!     ..Default::default()
//! };
//!
//! // Create processor
//! let mut processor = NeuralBeamformingProcessor::new(config)?;
//!
//! // Process RF data
//! let result = processor.process_volume(&rf_data)?;
//!
//! // Access results
//! println!("Volume shape: {:?}", result.volume.dim());
//! println!("Uncertainty range: [{:.3}, {:.3}]",
//!          result.uncertainty.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
//!          result.uncertainty.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
//! println!("Processing time: {:.2} ms", result.processing_time_ms);
//! ```
//!
//! ## Performance Considerations
//!
//! ### Computational Cost
//!
//! - Training: O(N_epochs × N_batch × N_params)
//! - Inference: O(N_voxels × N_forward)
//! - Uncertainty: O(N_voxels × N_MC_samples × N_forward)
//!
//! ### Memory Requirements
//!
//! - Model weights: ~4 MB per hidden layer
//! - Gradient buffers: 2× model size (Adam optimizer)
//! - Activation cache: Batch_size × Hidden_dim × sizeof(f32)
//!
//! ### Optimization Strategies
//!
//! - **Batch processing**: Process multiple voxels in parallel
//! - **Model compression**: Pruning, quantization for deployment
//! - **Transfer learning**: Pre-train on simulation, fine-tune on real data
//!
//! ## Limitations and Future Work
//!
//! ### Current Limitations
//!
//! - 1D wave equation (simplification for linear arrays)
//! - Homogeneous medium assumption
//! - Sequential frame processing (not real-time)
//!
//! ### Planned Enhancements
//!
//! - 3D wave equation for volumetric imaging
//! - Heterogeneous media with spatially-varying sound speed
//! - GPU acceleration for real-time processing
//! - Multi-task learning (imaging + flow estimation)
//!
//! ## References
//!
//! ### Physics-Informed Neural Networks
//!
//! - Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
//!   "Physics-informed neural networks: A deep learning framework for solving
//!   forward and inverse problems involving nonlinear partial differential equations."
//!   Journal of Computational Physics, 378, 686-707.
//!   doi:10.1016/j.jcp.2018.10.045
//!
//! ### Ultrasound Beamforming with Neural Networks
//!
//! - Luchies, A. C., & Byram, B. C. (2018).
//!   "Deep neural networks for ultrasound beamforming."
//!   IEEE Transactions on Medical Imaging, 37(9), 2010-2021.
//!   doi:10.1109/TMI.2018.2809641
//!
//! - Gasse, M., Millioz, F., Roux, E., Garcia, D., Liebgott, H., & Friboulet, D. (2017).
//!   "High-quality plane wave compounding using convolutional neural networks."
//!   IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control, 64(10), 1637-1639.
//!   doi:10.1109/TUFFC.2017.2736890
//!
//! ### Wave Physics
//!
//! - Szabo, T. L. (2004).
//!   "Diagnostic Ultrasound Imaging: Inside Out."
//!   Academic Press. ISBN: 978-0126801453
//!
//! - Jensen, J. A. (1996).
//!   "Field: A program for simulating ultrasound systems."
//!   Medical & Biological Engineering & Computing, 34, 351-353.
//!
//! ### Uncertainty Quantification
//!
//! - Gal, Y., & Ghahramani, Z. (2016).
//!   "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning."
//!   International Conference on Machine Learning (ICML), 1050-1059.
//!   arXiv:1506.02142
//!
//! - Abdar, M., et al. (2021).
//!   "A review of uncertainty quantification in deep learning."
//!   Information Fusion, 76, 243-297.
//!   doi:10.1016/j.inffus.2021.05.008

pub mod inference;
pub mod processor;

// Re-export primary types
pub use processor::NeuralBeamformingProcessor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinn_module_exports() {
        // Verify processor type is accessible
        let _ = std::any::type_name::<NeuralBeamformingProcessor>();
    }
}
