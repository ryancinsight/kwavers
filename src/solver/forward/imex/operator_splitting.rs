//! Operator splitting strategies for IMEX schemes

use super::traits::OperatorSplitting;
use crate::core::error::KwaversResult;
use ndarray::Array3;

/// Lie-Trotter splitting (first-order)
#[derive(Debug, Clone)]
pub struct LieTrotterSplitting;

impl LieTrotterSplitting {
    /// Create a new Lie-Trotter splitting
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for LieTrotterSplitting {
    fn default() -> Self {
        Self::new()
    }
}

impl OperatorSplitting for LieTrotterSplitting {
    fn split_step<F, G>(
        &self,
        field: &Array3<f64>,
        dt: f64,
        operator_a: F,
        operator_b: G,
    ) -> KwaversResult<Array3<f64>>
    where
        F: Fn(&Array3<f64>, f64) -> KwaversResult<Array3<f64>>,
        G: Fn(&Array3<f64>, f64) -> KwaversResult<Array3<f64>>,
    {
        // Step 1: Apply operator A for full time step
        let intermediate = operator_a(field, dt)?;

        // Step 2: Apply operator B for full time step
        operator_b(&intermediate, dt)
    }

    fn order(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "Lie-Trotter"
    }
}

/// Strang splitting (second-order)
#[derive(Debug, Clone)]
pub struct StrangSplitting;

impl StrangSplitting {
    /// Create a new Strang splitting
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for StrangSplitting {
    fn default() -> Self {
        Self::new()
    }
}

impl OperatorSplitting for StrangSplitting {
    fn split_step<F, G>(
        &self,
        field: &Array3<f64>,
        dt: f64,
        operator_a: F,
        operator_b: G,
    ) -> KwaversResult<Array3<f64>>
    where
        F: Fn(&Array3<f64>, f64) -> KwaversResult<Array3<f64>>,
        G: Fn(&Array3<f64>, f64) -> KwaversResult<Array3<f64>>,
    {
        // Step 1: Apply operator A for half time step
        let step1 = operator_a(field, dt / 2.0)?;

        // Step 2: Apply operator B for full time step
        let step2 = operator_b(&step1, dt)?;

        // Step 3: Apply operator A for half time step
        operator_a(&step2, dt / 2.0)
    }

    fn order(&self) -> usize {
        2
    }

    fn name(&self) -> &str {
        "Strang"
    }
}

/// Yoshida splitting (fourth-order)
#[derive(Debug, Clone)]
pub struct YoshidaSplitting {
    /// Yoshida coefficients
    w0: f64,
    w1: f64,
}

impl YoshidaSplitting {
    /// Create a new Yoshida splitting
    #[must_use]
    pub fn new() -> Self {
        let w1 = 1.0 / (2.0 - 2.0_f64.powf(1.0 / 3.0));
        let w0 = 1.0 - 2.0 * w1;

        Self { w0, w1 }
    }
}

impl Default for YoshidaSplitting {
    fn default() -> Self {
        Self::new()
    }
}

impl OperatorSplitting for YoshidaSplitting {
    fn split_step<F, G>(
        &self,
        field: &Array3<f64>,
        dt: f64,
        operator_a: F,
        operator_b: G,
    ) -> KwaversResult<Array3<f64>>
    where
        F: Fn(&Array3<f64>, f64) -> KwaversResult<Array3<f64>>,
        G: Fn(&Array3<f64>, f64) -> KwaversResult<Array3<f64>>,
    {
        // Yoshida's 4th order splitting
        // S(w1*dt) S(w0*dt) S(w1*dt) where S is Strang splitting

        let strang = StrangSplitting::new();

        // First Strang step with w1*dt
        let step1 = strang.split_step(field, self.w1 * dt, &operator_a, &operator_b)?;

        // Second Strang step with w0*dt
        let step2 = strang.split_step(&step1, self.w0 * dt, &operator_a, &operator_b)?;

        // Third Strang step with w1*dt
        strang.split_step(&step2, self.w1 * dt, &operator_a, &operator_b)
    }

    fn order(&self) -> usize {
        4
    }

    fn name(&self) -> &str {
        "Yoshida"
    }
}

/// Recursive splitting for higher orders
#[derive(Debug, Clone)]
pub struct RecursiveSplitting {
    order: usize,
    coefficients: Vec<f64>,
}

impl RecursiveSplitting {
    /// Create a new recursive splitting of given order
    #[must_use]
    pub fn new(order: usize) -> Self {
        let coefficients = Self::compute_coefficients(order);
        Self {
            order,
            coefficients,
        }
    }

    /// Compute splitting coefficients for given order
    fn compute_coefficients(order: usize) -> Vec<f64> {
        match order {
            1 => vec![1.0],
            2 => vec![0.5, 0.5],
            4 => {
                // Yoshida coefficients
                let w1 = 1.0 / (2.0 - 2.0_f64.powf(1.0 / 3.0));
                let w0 = 1.0 - 2.0 * w1;
                vec![w1 / 2.0, w1 / 2.0, w0 / 2.0, w0 / 2.0, w1 / 2.0, w1 / 2.0]
            }
            6 => {
                // 6th order coefficients (Yoshida 1990)
                let w1 = 1.0 / (2.0 - 2.0_f64.powf(1.0 / 5.0));
                let w0 = 1.0 - 2.0 * w1;
                vec![w1 / 2.0, w1 / 2.0, w0 / 2.0, w0 / 2.0, w1 / 2.0, w1 / 2.0]
            }
            _ => {
                // Default to Strang for unsupported orders
                vec![0.5, 0.5]
            }
        }
    }
}

impl OperatorSplitting for RecursiveSplitting {
    fn split_step<F, G>(
        &self,
        field: &Array3<f64>,
        dt: f64,
        operator_a: F,
        operator_b: G,
    ) -> KwaversResult<Array3<f64>>
    where
        F: Fn(&Array3<f64>, f64) -> KwaversResult<Array3<f64>>,
        G: Fn(&Array3<f64>, f64) -> KwaversResult<Array3<f64>>,
    {
        let mut result = field.clone();

        // Apply splitting based on coefficients
        let n = self.coefficients.len();
        for i in 0..n {
            if i % 2 == 0 {
                // Apply operator A
                result = operator_a(&result, self.coefficients[i] * dt)?;
            } else {
                // Apply operator B
                result = operator_b(&result, self.coefficients[i] * dt)?;
            }
        }

        Ok(result)
    }

    fn order(&self) -> usize {
        self.order
    }

    fn name(&self) -> &str {
        "Recursive"
    }
}
