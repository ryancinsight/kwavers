//! Linear inference engine for single-layer neural network models.
//!
//! [`InferenceEngine`] implements a single affine (linear) layer:
//!
//! ```text
//! Y = X · W + b
//! ```
//!
//! where
//! - `X` has shape `(n_samples, input_dim)`,
//! - `W` has shape `(input_dim, output_dim)`,
//! - `b` has shape `(output_dim,)` (optional),
//! - `Y` has shape `(n_samples, output_dim)`.
//!
//! When `normalize_output` is `true`, each output row is divided by its L2 norm
//! (producing unit-length direction vectors — useful as logit pre-processing).

use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use ndarray::{Array1, Array2};

/// Single affine-layer inference engine.
///
/// Wraps a weight matrix and optional bias vector.  `forward` performs the
/// matrix multiply plus optional row-wise L2 normalisation in a single pass.
#[derive(Debug)]
pub struct InferenceEngine {
    /// Weight matrix of shape `(input_dim, output_dim)`.
    weights: Array2<f32>,
    /// Optional bias vector of length `output_dim`.
    bias: Option<Array1<f32>>,
    /// Number of samples processed per call (informational; does not change output).
    batch_size: usize,
    /// When `true`, each output row is normalised to unit L2 norm.
    normalize_output: bool,
}

impl InferenceEngine {
    /// Construct from an explicit weight matrix and optional bias.
    ///
    /// # Arguments
    /// - `weights` — shape `(input_dim, output_dim)`.
    /// - `bias` — optional `(output_dim,)` additive term.
    /// - `batch_size` — preferred mini-batch size (informational).
    /// - `normalize_output` — if `true`, output rows are L2-normalised.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn from_weights(
        weights: Array2<f32>,
        bias: Option<Array1<f32>>,
        batch_size: usize,
        normalize_output: bool,
    ) -> Self {
        Self {
            weights,
            bias,
            batch_size,
            normalize_output,
        }
    }

    /// Run the affine transform on `input`.
    ///
    /// # Errors
    /// Returns [`KwaversError::Validation`] when `input.ncols() != weights.nrows()`.
    pub fn forward(&self, input: &Array2<f32>) -> KwaversResult<Array2<f32>> {
        let (_, input_dim) = input.dim();
        let (weight_in, _output_dim) = self.weights.dim();

        if input_dim != weight_in {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "input".to_owned(),
                value: format!("ncols={input_dim}"),
                constraint: format!("must equal weights.nrows()={weight_in}"),
            }));
        }

        // Y = X · W
        let mut output = input.dot(&self.weights);

        // Y += b  (broadcast)
        if let Some(ref b) = self.bias {
            for mut row in output.rows_mut() {
                row += b;
            }
        }

        // Optional row-wise L2 normalisation
        if self.normalize_output {
            for mut row in output.rows_mut() {
                let norm = row.mapv(|x| x * x).sum().sqrt();
                if norm > f32::EPSILON {
                    row /= norm;
                }
            }
        }

        Ok(output)
    }

    /// Returns the expected number of input features (`weights.nrows()`).
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.weights.nrows()
    }

    /// Returns the number of output units (`weights.ncols()`).
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.weights.ncols()
    }

    /// Returns the preferred mini-batch size.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    // Identity weight matrix: Y = X · I = X (no bias).
    // Expected output == input, verifiable analytically.
    #[test]
    fn test_forward_identity_transform() {
        let weights = Array2::eye(3); // (3, 3) identity
        let engine = InferenceEngine::from_weights(weights, None, 32, false);

        let input = array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let output = engine.forward(&input).unwrap();

        assert_eq!(output.shape(), &[2, 3]);
        // Each output row must equal the corresponding input row exactly.
        assert!((output[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((output[[0, 1]] - 2.0).abs() < 1e-6);
        assert!((output[[0, 2]] - 3.0).abs() < 1e-6);
        assert!((output[[1, 0]] - 4.0).abs() < 1e-6);
    }

    // Bias-only transform: W = I, b = [10, 20, 30].
    // Expected: Y[i,j] = X[i,j] + b[j].
    #[test]
    fn test_forward_with_bias() {
        let weights = Array2::eye(3);
        let bias = ndarray::array![10.0_f32, 20.0, 30.0];
        let engine = InferenceEngine::from_weights(weights, Some(bias), 32, false);

        let input = array![[1.0_f32, 1.0, 1.0]];
        let output = engine.forward(&input).unwrap();

        assert!((output[[0, 0]] - 11.0).abs() < 1e-6);
        assert!((output[[0, 1]] - 21.0).abs() < 1e-6);
        assert!((output[[0, 2]] - 31.0).abs() < 1e-6);
    }

    // Dimension mismatch must produce a Validation error (not panic).
    #[test]
    fn test_forward_dimension_mismatch_returns_error() {
        let weights = Array2::<f32>::eye(4); // expects 4-column input
        let engine = InferenceEngine::from_weights(weights, None, 32, false);

        let input = array![[1.0_f32, 2.0, 3.0]]; // 3 columns — wrong
        let result = engine.forward(&input);

        assert!(
            matches!(result, Err(KwaversError::Validation(_))),
            "expected Validation error, got {result:?}"
        );
    }

    // Normalised output: rows of Y must have unit L2 norm.
    #[test]
    fn test_forward_normalised_output_unit_norm() {
        let weights = Array2::eye(3);
        let engine = InferenceEngine::from_weights(weights, None, 32, true);

        let input = array![[3.0_f32, 4.0, 0.0]]; // norm = 5.0
        let output = engine.forward(&input).unwrap();

        let norm: f32 = output.row(0).mapv(|x| x * x).sum().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expected unit norm, got {norm}");
        // Values must be 3/5, 4/5, 0
        assert!((output[[0, 0]] - 0.6).abs() < 1e-5);
        assert!((output[[0, 1]] - 0.8).abs() < 1e-5);
        assert!((output[[0, 2]]).abs() < 1e-5);
    }

    // Dimension accessors must reflect the constructed shape.
    #[test]
    fn test_dimension_accessors() {
        let weights = Array2::<f32>::zeros((128, 10));
        let engine = InferenceEngine::from_weights(weights, None, 64, false);

        assert_eq!(engine.input_dim(), 128);
        assert_eq!(engine.output_dim(), 10);
        assert_eq!(engine.batch_size(), 64);
    }
}
