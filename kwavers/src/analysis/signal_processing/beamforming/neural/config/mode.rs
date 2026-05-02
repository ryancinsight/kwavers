/// Neural beamforming processing modes.
///
/// Different modes trade off between computational cost, image quality,
/// and physical consistency.
///
/// ## Mode Selection Guidelines
///
/// - **NeuralOnly**: Fastest, purely data-driven, may violate physics
/// - **Hybrid**: Good balance, combines traditional + neural refinement
/// - **PhysicsInformed**: Best quality, enforces wave equation constraints
/// - **Adaptive**: Robust, switches based on signal quality metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NeuralBeamformingMode {
    /// Pure neural network beamforming without traditional preprocessing.
    NeuralOnly,

    /// Hybrid approach: traditional delay-and-sum followed by neural refinement.
    #[default]
    Hybrid,

    /// Physics-informed neural networks (PINN) enforcing wave equation.
    ///
    /// **Requires**: `pinn` feature flag
    #[cfg(feature = "pinn")]
    PhysicsInformed,

    /// Adaptive mode switching based on real-time signal quality assessment.
    Adaptive,
}
