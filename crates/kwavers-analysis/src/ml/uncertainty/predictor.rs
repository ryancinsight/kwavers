//! Solver-agnostic prediction contract for PINN uncertainty estimation.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{
    Array1,
    Array2,
};

/// PINN prediction surface consumed by analysis-side uncertainty estimators.
pub trait PinnUncertaintyPredictor {
    /// Predict output values for normalized coordinate vectors.
    ///
    /// # Errors
    /// - Propagates backend/model prediction failures.
    fn predict_coordinates(&self, x: &Array1<f64>, t: &Array1<f64>) -> KwaversResult<Array2<f32>>;

    /// Predict output values from a two-column `[x, t]` input matrix.
    ///
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] when fewer than two columns are
    ///   provided.
    /// - Propagates backend/model prediction failures.
    fn predict_inputs(&self, inputs: &Array2<f32>) -> KwaversResult<Array2<f32>> {
        if inputs.ncols() < 2 {
            return Err(KwaversError::InvalidInput(
                "PINN uncertainty inputs must contain x and t columns".to_owned(),
            ));
        }

        let x = inputs.column(0).mapv(f64::from).to_owned();
        let t = inputs.column(1).mapv(f64::from).to_owned();
        self.predict_coordinates(&x, &t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use array;

    struct SumPredictor;

    impl PinnUncertaintyPredictor for SumPredictor {
        fn predict_coordinates(
            &self,
            x: &Array1<f64>,
            t: &Array1<f64>,
        ) -> KwaversResult<Array2<f32>> {
            let mut result = Array2::zeros((x.len(), 1));
            for idx in 0..x.len() {
                result[[idx, 0]] = (x[idx] + t[idx]) as f32;
            }
            Ok(result)
        }
    }

    #[test]
    fn input_matrix_routes_columns_to_predictor() {
        let inputs = array![[1.0_f32, 0.25_f32], [2.0_f32, 0.5_f32]];

        let prediction = SumPredictor.predict_inputs(&inputs).unwrap();

        assert_eq!(prediction.dim(), (2, 1));
        assert_eq!(prediction[[0, 0]], 1.25);
        assert_eq!(prediction[[1, 0]], 2.5);
    }

    #[test]
    fn input_matrix_rejects_missing_time_column() {
        let inputs = Array2::from_elem((2, 1), 1.0_f32);

        let err = SumPredictor.predict_inputs(&inputs).unwrap_err();
        let msg = format!("{err:?}");

        assert!(
            msg.contains("x and t columns"),
            "invalid input error must name the missing coordinate contract; got: {msg}"
        );
    }
}
