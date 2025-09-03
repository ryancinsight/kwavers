//! Beamforming algorithms

use ndarray::{Array1, Array2};

/// Beamforming algorithm types with literature-based implementations
#[derive(Debug, Clone)]
pub enum BeamformingAlgorithm {
    /// Delay-and-Sum (conventional beamforming)
    DelaySum,
    /// Minimum Variance Distortionless Response (MVDR/Capon)
    MVDR {
        diagonal_loading: f64,
        spatial_smoothing: bool,
    },
    /// `MUltiple` `SIgnal` Classification
    MUSIC {
        signal_subspace_dimension: usize,
        spatial_smoothing: bool,
    },
    /// Capon Beamforming with Regularization
    CaponRegularized {
        diagonal_loading: f64,
        uncertainty_set_size: f64,
    },
    /// Linearly Constrained Minimum Variance (LCMV)
    LCMV {
        constraint_matrix: Array2<f64>,
        response_vector: Array1<f64>,
    },
    /// Generalized Sidelobe Canceller (GSC)
    GSC {
        main_beam_weight: f64,
        adaptation_step_size: f64,
    },
    /// Compressive Beamforming
    Compressive {
        sparsity_parameter: f64,
        dictionary_size: usize,
    },
}

/// Trait for algorithm implementations
pub trait AlgorithmImplementation {
    fn process(&self, data: &Array2<f64>) -> Array1<f64>;
}
