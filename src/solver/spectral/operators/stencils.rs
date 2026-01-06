//! Finite difference stencil computations
//!
//! Implements high-order finite difference stencils for spatial derivatives
//! Based on: Fornberg, B. (1988). "Generation of finite difference formulas on
//! arbitrarily spaced grids." Mathematics of computation, 51(184), 699-706.

use ndarray::Array1;

use crate::error::{KwaversError, ValidationError};

/// Stencil weights for finite differences
#[derive(Debug, Clone)]
pub struct StencilWeights {
    pub forward: Array1<f64>,
    pub backward: Array1<f64>,
    pub central: Array1<f64>,
}

/// Compute derivative stencils for given order and accuracy
pub fn compute_derivative_stencils(
    order: usize,
    accuracy: usize,
) -> Result<StencilWeights, KwaversError> {
    match (order, accuracy) {
        (1, 2) => Ok(second_order_first_derivative()),
        (1, 4) => Ok(fourth_order_first_derivative()),
        (2, 2) => Ok(second_order_second_derivative()),
        (2, 4) => Ok(fourth_order_second_derivative()),
        _ => Err(KwaversError::Validation(ValidationError::FieldValidation {
            field: "stencil_config".to_string(),
            value: format!("order={}, accuracy={}", order, accuracy),
            constraint: "Supported combinations: (1,2), (1,4), (2,2), (2,4)".to_string(),
        })),
    }
}

fn second_order_first_derivative() -> StencilWeights {
    StencilWeights {
        forward: Array1::from(vec![-1.0, 1.0]),
        backward: Array1::from(vec![-1.0, 1.0]),
        central: Array1::from(vec![-0.5, 0.0, 0.5]),
    }
}

fn fourth_order_first_derivative() -> StencilWeights {
    StencilWeights {
        forward: Array1::from(vec![-25.0 / 12.0, 4.0, -3.0, 4.0 / 3.0, -1.0 / 4.0]),
        backward: Array1::from(vec![1.0 / 4.0, -4.0 / 3.0, 3.0, -4.0, 25.0 / 12.0]),
        central: Array1::from(vec![1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0]),
    }
}

fn second_order_second_derivative() -> StencilWeights {
    StencilWeights {
        forward: Array1::from(vec![1.0, -2.0, 1.0]),
        backward: Array1::from(vec![1.0, -2.0, 1.0]),
        central: Array1::from(vec![1.0, -2.0, 1.0]),
    }
}

fn fourth_order_second_derivative() -> StencilWeights {
    StencilWeights {
        forward: Array1::from(vec![
            15.0 / 4.0,
            -77.0 / 6.0,
            107.0 / 6.0,
            -13.0,
            61.0 / 12.0,
            -5.0 / 6.0,
        ]),
        backward: Array1::from(vec![
            -5.0 / 6.0,
            61.0 / 12.0,
            -13.0,
            107.0 / 6.0,
            -77.0 / 6.0,
            15.0 / 4.0,
        ]),
        central: Array1::from(vec![
            -1.0 / 12.0,
            4.0 / 3.0,
            -5.0 / 2.0,
            4.0 / 3.0,
            -1.0 / 12.0,
        ]),
    }
}
