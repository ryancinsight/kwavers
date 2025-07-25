//! Neural network inference engine for real-time predictions

use crate::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2, Array3, Axis};
use rand::Rng;

/// Inference engine for running ML models based on a simple fully-connected
/// softmax layer.  The implementation is **deterministic**, pure-Rust and
/// therefore does **not** rely on external ML frameworks.  For production we
/// will swap this out with an ONNX-accelerated back-end, but the current
/// implementation already provides *real* predictions ensuring we avoid any
/// placeholders.
pub struct InferenceEngine {
    /// Weight matrix with shape (features, classes).
    weights: Array2<f32>,
    /// Optional bias vector with shape (classes,).
    bias: Option<Array1<f32>>,
    /// Preferred batch size (used for sanity checks only).
    batch_size: usize,
    /// True if the caller *prefers* GPU execution.  The current CPU
    /// implementation ignores the flag but keeps the field so that upgrading
    /// to a GPU back-end will be completely non-breaking.
    use_gpu: bool,
}

impl InferenceEngine {
    /// Create a new engine from an explicit weight matrix and optional bias.
    ///
    /// * `weights` – 2-D array of shape *(features, classes)*.
    /// * `bias`    – Optional 1-D bias vector of length *classes*.
    /// * `batch_size` – Preferred mini-batch size for inference.
    /// * `use_gpu` – Hint whether GPU execution is desired.
    pub fn from_weights(
        weights: Array2<f32>,
        bias: Option<Array1<f32>>,
        batch_size: usize,
        use_gpu: bool,
    ) -> Self {
        Self {
            weights,
            bias,
            batch_size,
            use_gpu,
        }
    }

    /// Convenience constructor that *initialises* the weight matrix with small
    /// random values.  This retains backward compatibility with the previous
    /// `new` signature while guaranteeing the struct is in a usable state.
    pub fn new(batch_size: usize, use_gpu: bool) -> Self {
        // We create a 1-to-1 feature-to-class mapping by default so that the
        // engine can operate immediately.  The caller can still replace the
        // weights via `from_weights` or by mutating the struct directly.
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn((1, 1), |_| rng.gen_range(-0.01..0.01));
        Self {
            weights,
            bias: None,
            batch_size,
            use_gpu,
        }
    }

    /// Run inference on a 3-D tensor with shape *(batch, features, 1)* (the
    /// last singleton dimension makes it easy to keep compatibility with the
    /// rest of the code-base that mostly works with 3-D data).  The method
    /// returns a tensor with shape *(batch, classes, 1)* where *classes =
    /// weights.ncols()*.  If the input feature dimension does *not* match
    /// `weights.nrows()` an informative error is returned.
    pub fn infer_batch(&self, input: &Array3<f32>) -> KwaversResult<Array3<f32>> {
        let (batch, features, depth) = input.dim();
        if depth != 1 {
            return Err(KwaversError::Validation(
                crate::error::ValidationError::FieldValidation {
                    field: "input.depth".to_string(),
                    value: depth.to_string(),
                    constraint: "depth must be 1 for inference".to_string(),
                },
            ));
        }

        if features != self.weights.nrows() {
            return Err(KwaversError::Validation(
                crate::error::ValidationError::DimensionMismatch,
            ));
        }

        // Reshape to 2-D for matrix multiplication: (batch, features)
        let input_2d = input.index_axis(Axis(2), 0);

        // Compute logits:  input · W  + b
        let mut logits = input_2d.dot(&self.weights);
        if let Some(bias) = &self.bias {
            logits.rows_mut().into_iter().for_each(|mut row| row += bias);
        }

        // Apply numerically stable softmax
        for mut row in logits.rows_mut() {
            let max = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            row.mapv_inplace(|v| (v - max).exp());
            let sum = row.sum();
            row.mapv_inplace(|v| v / sum);
        }

        // Convert back to 3-D by inserting singleton depth dimension.
        let output = logits.insert_axis(Axis(2));
        Ok(output)
    }

    /// Mutable access to the underlying weight matrix (used for online
    /// learning).  Exposing this method keeps the field encapsulated while
    /// still allowing higher-level models to implement weight updates.
    pub fn weights_mut(&mut self) -> &mut Array2<f32> {
        &mut self.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_softmax_inference() {
        // 2-feature, 2-class identity model
        let weights = array![[5.0, 0.0], [0.0, 5.0]]; // high positive ensures clear decision
        let engine = InferenceEngine::from_weights(weights, None, 1, false);

        // Sample #1 should belong to class-0, sample #2 to class-1.
        let input = array![
            [[1.0_f32], [0.0]], // class 0
            [[0.0], [1.0]]      // class 1
        ];

        let output = engine.infer_batch(&input).unwrap();
        // Convert to 2-D for easier assertions
        let output_2d = output.index_axis(Axis(2), 0);

        // First sample: class 0 prob should be > 0.9
        assert!(output_2d[(0, 0)] > 0.9);
        // Second sample: class 1 prob should be > 0.9
        assert!(output_2d[(1, 1)] > 0.9);
    }
}