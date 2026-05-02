/// Physics constraint parameters for neural beamforming.
///
/// Controls the strength of various physics-based regularization terms.
///
/// ## Constraint Types
///
/// - **Reciprocity**: Time-reversal symmetry H(A→B) = H(B→A)
/// - **Coherence**: Spatial smoothness ∇²I
/// - **Sparsity**: Focused point-spread function via L1 penalty
#[derive(Debug, Clone)]
pub struct PhysicsParameters {
    /// Weight for reciprocity constraint (time-reversal symmetry). Range: [0.0, 10.0]
    pub reciprocity_weight: f64,

    /// Weight for coherence constraint (spatial smoothness). Range: [0.0, 5.0]
    pub coherence_weight: f64,

    /// Weight for sparsity constraint (L1 regularization). Range: [0.0, 1.0]
    pub sparsity_weight: f64,

    /// Coherence diffusion coefficient for smoothing. Range: [0.0, 1.0]
    pub diffusion_coefficient: f64,

    /// Sparsity soft threshold for L1 penalty. Range: [0.0, 1.0]
    pub soft_threshold: f64,
}

impl Default for PhysicsParameters {
    fn default() -> Self {
        Self {
            reciprocity_weight: 1.0,
            coherence_weight: 0.5,
            sparsity_weight: 0.1,
            diffusion_coefficient: 0.01,
            soft_threshold: 0.05,
        }
    }
}
