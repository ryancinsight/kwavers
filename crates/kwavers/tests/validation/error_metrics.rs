//! Error Metrics for Validation Testing
//!
//! This module provides rigorous error metric computations for comparing
//! numerical solutions against analytical ground truth.
//!
//! # Supported Norms
//!
//! 1. **L² norm (Root Mean Square Error)**:
//!    ```text
//!    ||e||₂ = √(∑ᵢ(computed_i - analytical_i)² / N)
//!    ```
//!
//! 2. **L∞ norm (Maximum Absolute Error)**:
//!    ```text
//!    ||e||∞ = max_i |computed_i - analytical_i|
//!    ```
//!
//! 3. **Relative Error**:
//!    ```text
//!    relative_error = ||computed - analytical|| / ||analytical||
//!    ```
//!
//! # Design Principles
//!
//! - **Numerical Stability**: Avoid overflow/underflow in norm computations
//! - **Dimensional Analysis**: Check dimension compatibility before comparison
//! - **Zero Handling**: Graceful handling of zero analytical solutions
//! - **NaN/Inf Detection**: Report non-finite values as errors

/// Error metrics between computed and analytical solutions
#[derive(Debug, Clone, Copy)]
pub struct ErrorMetrics {
    /// L² norm: ||computed - analytical||₂
    pub l2_error: f64,
    /// L∞ norm: max|computed - analytical|
    pub linf_error: f64,
    /// Relative L² error: ||error||₂ / ||analytical||₂
    pub relative_l2_error: f64,
    /// Number of sample points
    pub n_points: usize,
}

#[allow(dead_code)]
impl ErrorMetrics {
    /// Compute error metrics between two vector fields
    ///
    /// # Arguments
    ///
    /// * `computed` - Numerically computed values
    /// * `analytical` - Analytical ground truth values
    ///
    /// # Returns
    ///
    /// Error metrics structure with L² and L∞ norms
    ///
    /// # Panics
    ///
    /// Panics if vectors have different lengths
    pub fn compute(computed: &[f64], analytical: &[f64]) -> Self {
        assert_eq!(
            computed.len(),
            analytical.len(),
            "Computed and analytical vectors must have same length"
        );

        let n = computed.len();
        if n == 0 {
            return Self {
                l2_error: 0.0,
                linf_error: 0.0,
                relative_l2_error: 0.0,
                n_points: 0,
            };
        }

        // Compute squared differences
        let mut sum_squared_error = 0.0_f64;
        let mut sum_squared_analytical = 0.0_f64;
        let mut max_error = 0.0_f64;

        for i in 0..n {
            let error = computed[i] - analytical[i];
            let abs_error = error.abs();

            // Check for non-finite values
            if !error.is_finite() || !analytical[i].is_finite() {
                return Self {
                    l2_error: f64::INFINITY,
                    linf_error: f64::INFINITY,
                    relative_l2_error: f64::INFINITY,
                    n_points: n,
                };
            }

            sum_squared_error += error * error;
            sum_squared_analytical += analytical[i] * analytical[i];
            max_error = max_error.max(abs_error);
        }

        // L² norm: RMS error
        let l2_error = (sum_squared_error / n as f64).sqrt();

        // Relative L² error
        let analytical_norm = (sum_squared_analytical / n as f64).sqrt();
        let relative_l2_error = if analytical_norm > 1e-15 {
            l2_error / analytical_norm
        } else {
            // Analytical solution is near zero - report absolute error
            l2_error
        };

        Self {
            l2_error,
            linf_error: max_error,
            relative_l2_error,
            n_points: n,
        }
    }

    /// Compute error metrics for multi-component vector fields (e.g., displacement)
    ///
    /// # Arguments
    ///
    /// * `computed` - Computed field as flat array [u0_x, u0_y, u1_x, u1_y, ...]
    /// * `analytical` - Analytical field with same layout
    /// * `n_points` - Number of spatial points
    /// * `n_components` - Number of components per point (e.g., 2 for 2D displacement)
    ///
    /// # Returns
    ///
    /// Error metrics treating the field as a single vector
    pub fn compute_field(
        computed: &[f64],
        analytical: &[f64],
        n_points: usize,
        n_components: usize,
    ) -> Self {
        assert_eq!(
            computed.len(),
            n_points * n_components,
            "Computed array size must be n_points × n_components"
        );
        assert_eq!(
            analytical.len(),
            n_points * n_components,
            "Analytical array size must be n_points × n_components"
        );

        Self::compute(computed, analytical)
    }

    /// Check if error is below tolerance
    pub fn within_tolerance(&self, tolerance: f64) -> bool {
        self.l2_error <= tolerance && self.linf_error <= tolerance
    }

    /// Check if relative error is below tolerance
    pub fn relative_within_tolerance(&self, tolerance: f64) -> bool {
        self.relative_l2_error <= tolerance
    }
}

/// Compute L² norm of a vector
///
/// # Mathematical Specification
///
/// ||v||₂ = √(∑ᵢ vᵢ² / N)
pub fn l2_norm(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }

    let sum_squared: f64 = v.iter().map(|&x| x * x).sum();
    (sum_squared / v.len() as f64).sqrt()
}

/// Compute L∞ norm of a vector
///
/// # Mathematical Specification
///
/// ||v||∞ = max_i |vᵢ|
pub fn linf_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x.abs()).fold(0.0, f64::max)
}

/// Compute relative error: ||computed - analytical|| / ||analytical||
///
/// # Returns
///
/// Relative L² error, or absolute error if analytical norm is near zero
pub fn relative_error(computed: &[f64], analytical: &[f64]) -> f64 {
    assert_eq!(computed.len(), analytical.len());

    let error = ErrorMetrics::compute(computed, analytical);
    error.relative_l2_error
}

/// Compute pointwise absolute errors
///
/// # Returns
///
/// Vector of |computed_i - analytical_i|
pub fn pointwise_errors(computed: &[f64], analytical: &[f64]) -> Vec<f64> {
    assert_eq!(computed.len(), analytical.len());

    computed
        .iter()
        .zip(analytical.iter())
        .map(|(&c, &a)| (c - a).abs())
        .collect()
}

/// Compute pointwise relative errors
///
/// # Returns
///
/// Vector of |computed_i - analytical_i| / max(|analytical_i|, ε)
/// where ε = 1e-15 prevents division by zero
pub fn pointwise_relative_errors(computed: &[f64], analytical: &[f64]) -> Vec<f64> {
    assert_eq!(computed.len(), analytical.len());

    const EPSILON: f64 = 1e-15;

    computed
        .iter()
        .zip(analytical.iter())
        .map(|(&c, &a)| {
            let error = (c - a).abs();
            let denom = a.abs().max(EPSILON);
            error / denom
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_norm() {
        let v = vec![3.0, 4.0];
        let norm = l2_norm(&v);
        // √((9 + 16) / 2) = √12.5 = 3.5355...
        assert!((norm - 3.5355339).abs() < 1e-6);
    }

    #[test]
    fn test_linf_norm() {
        let v = vec![1.0, -5.0, 3.0];
        let norm = linf_norm(&v);
        assert_eq!(norm, 5.0);
    }

    #[test]
    fn test_error_metrics_exact() {
        let computed = vec![1.0, 2.0, 3.0];
        let analytical = vec![1.0, 2.0, 3.0];

        let metrics = ErrorMetrics::compute(&computed, &analytical);

        assert_eq!(metrics.l2_error, 0.0);
        assert_eq!(metrics.linf_error, 0.0);
        assert_eq!(metrics.relative_l2_error, 0.0);
    }

    #[test]
    fn test_error_metrics_nonzero() {
        let computed = vec![1.0, 2.0, 3.0];
        let analytical = vec![1.0, 2.1, 3.0];

        let metrics = ErrorMetrics::compute(&computed, &analytical);

        // L² error: √((0 + 0.01 + 0) / 3) = √(0.01/3) ≈ 0.0577
        assert!((metrics.l2_error - 0.0577).abs() < 1e-3);
        assert!((metrics.linf_error - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_error_metrics_within_tolerance() {
        let computed = vec![1.0, 2.0, 3.0];
        let analytical = vec![1.001, 2.001, 3.001];

        let metrics = ErrorMetrics::compute(&computed, &analytical);

        assert!(metrics.within_tolerance(0.01));
        assert!(!metrics.within_tolerance(0.0001));
    }

    #[test]
    fn test_pointwise_errors() {
        let computed = vec![1.0, 2.0, 3.0];
        let analytical = vec![1.1, 1.9, 3.2];

        let errors = pointwise_errors(&computed, &analytical);

        assert!((errors[0] - 0.1).abs() < 1e-10);
        assert!((errors[1] - 0.1).abs() < 1e-10);
        assert!((errors[2] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_relative_error_nonzero_baseline() {
        let computed = vec![10.0, 20.0];
        let analytical = vec![10.0, 22.0];

        let rel_err = relative_error(&computed, &analytical);

        // ||error||₂ = √(4/2) = √2
        // ||analytical||₂ = √(484/2) = √242
        // relative = √2 / √242 ≈ 0.0828
        assert!((rel_err - 0.0828).abs() < 1e-3);
    }

    #[test]
    fn test_pointwise_relative_errors() {
        let computed = vec![10.0, 20.0];
        let analytical = vec![10.0, 22.0];

        let rel_errors = pointwise_relative_errors(&computed, &analytical);

        assert_eq!(rel_errors[0], 0.0);
        assert!((rel_errors[1] - 2.0 / 22.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "must have same length")]
    fn test_error_metrics_dimension_mismatch() {
        let computed = vec![1.0, 2.0];
        let analytical = vec![1.0, 2.0, 3.0];

        let _ = ErrorMetrics::compute(&computed, &analytical);
    }

    #[test]
    fn test_error_metrics_infinite_value() {
        let computed = vec![1.0, f64::INFINITY, 3.0];
        let analytical = vec![1.0, 2.0, 3.0];

        let metrics = ErrorMetrics::compute(&computed, &analytical);

        assert!(metrics.l2_error.is_infinite());
        assert!(metrics.linf_error.is_infinite());
    }

    #[test]
    fn test_error_metrics_field() {
        // 3 points, 2 components each: [u0, v0, u1, v1, u2, v2]
        let computed = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let analytical = vec![1.1, 2.1, 3.0, 4.0, 5.1, 6.1];

        let metrics = ErrorMetrics::compute_field(&computed, &analytical, 3, 2);

        assert_eq!(metrics.n_points, 6);
        assert!(metrics.l2_error > 0.0);
    }
}
