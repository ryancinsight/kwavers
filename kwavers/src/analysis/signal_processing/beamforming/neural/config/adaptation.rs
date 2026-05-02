/// Adaptation parameters for learning and feedback.
///
/// Controls how the neural network adapts based on performance feedback
/// and signal quality metrics.
#[derive(Debug, Clone)]
pub struct AdaptationParameters {
    /// Learning rate for online adaptation. Range: [1e-6, 1e-2]
    pub learning_rate: f64,

    /// Uncertainty threshold for mode switching (adaptive mode). Range: [0.0, 1.0]
    pub uncertainty_threshold: f64,

    /// Signal quality threshold for mode selection (coherence factor). Range: [0.0, 1.0]
    pub quality_threshold: f64,

    /// Enable online learning during processing.
    pub enable_online_learning: bool,

    /// Number of adaptation iterations per feedback cycle. Range: [1, 100]
    pub adaptation_iterations: usize,

    /// Momentum coefficient for gradient descent. Range: [0.0, 0.99]
    pub momentum: f64,
}

impl Default for AdaptationParameters {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            uncertainty_threshold: 0.3,
            quality_threshold: 0.7,
            enable_online_learning: false,
            adaptation_iterations: 10,
            momentum: 0.9,
        }
    }
}
