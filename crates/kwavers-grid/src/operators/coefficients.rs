//! Finite difference coefficients module
//!
//! This module owns the mathematical knowledge of finite difference coefficients
//! following the Information Expert GRASP principle.
//!
//! References:
//! - Fornberg, B. (1988). "Generation of finite difference formulas"
//! - LeVeque, R.J. (2007). "Finite Difference Methods for ODEs and PDEs"

use eunomia::FloatElement;

/// Spatial accuracy order for finite difference schemes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FdAccuracyOrder {
    /// Second-order accurate (3-point stencil)
    Second,
    /// Fourth-order accurate (5-point stencil)
    Fourth,
    /// Sixth-order accurate (7-point stencil)
    Sixth,
    /// Eighth-order accurate (9-point stencil)
    Eighth,
}

/// Finite difference coefficients for different orders
#[derive(Debug)]
pub struct FDCoefficients;

impl FDCoefficients {
    /// Get coefficients for first derivative (generic over float type)
    ///
    /// Coefficients are converted once from their `f64` rational evaluation to
    /// the native precision selected by `T`.
    #[must_use]
    pub fn first_derivative<T: FloatElement>(order: FdAccuracyOrder) -> Vec<T> {
        match order {
            FdAccuracyOrder::Second => vec![T::from_f64(0.5)],
            FdAccuracyOrder::Fourth => {
                vec![T::from_f64(-1.0 / 12.0), T::from_f64(2.0 / 3.0)]
            }
            FdAccuracyOrder::Sixth => {
                vec![
                    T::from_f64(1.0 / 60.0),
                    T::from_f64(-3.0 / 20.0),
                    T::from_f64(3.0 / 4.0),
                ]
            }
            FdAccuracyOrder::Eighth => {
                vec![
                    T::from_f64(-1.0 / 280.0),
                    T::from_f64(4.0 / 105.0),
                    T::from_f64(-1.0 / 5.0),
                    T::from_f64(4.0 / 5.0),
                ]
            }
        }
    }

    /// Get coefficients for second derivative pairs (off-center points)
    ///
    /// Returns coefficients for the points at ±h, ±2h, etc. from center.
    /// The center point coefficient should be computed separately.
    ///
    /// Coefficients are rounded once to the native precision selected by `T`.
    #[must_use]
    pub fn second_derivative_pairs<T: FloatElement>(order: FdAccuracyOrder) -> Vec<T> {
        match order {
            FdAccuracyOrder::Second => vec![T::from_f64(1.0)],
            FdAccuracyOrder::Fourth => {
                vec![T::from_f64(-1.0 / 12.0), T::from_f64(4.0 / 3.0)]
            }
            FdAccuracyOrder::Sixth => {
                vec![
                    T::from_f64(1.0 / 90.0),
                    T::from_f64(-3.0 / 20.0),
                    T::from_f64(3.0 / 2.0),
                ]
            }
            FdAccuracyOrder::Eighth => {
                vec![
                    T::from_f64(-1.0 / 560.0),
                    T::from_f64(8.0 / 315.0),
                    T::from_f64(-1.0 / 5.0),
                    T::from_f64(8.0 / 5.0),
                ]
            }
        }
    }

    /// Get center coefficient for second derivative
    ///
    /// The coefficient is rounded once to the native precision selected by `T`.
    #[must_use]
    pub fn second_derivative_center<T: FloatElement>(order: FdAccuracyOrder) -> T {
        match order {
            FdAccuracyOrder::Second => T::from_f64(-2.0),
            FdAccuracyOrder::Fourth => T::from_f64(-5.0 / 2.0),
            FdAccuracyOrder::Sixth => T::from_f64(-49.0 / 18.0),
            FdAccuracyOrder::Eighth => T::from_f64(-205.0 / 72.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_first_derivative_coefficients() {
        let coeffs: Vec<f64> = FDCoefficients::first_derivative(FdAccuracyOrder::Second);
        assert_eq!(coeffs.len(), 1);
        assert_relative_eq!(coeffs[0], 0.5, epsilon = 1e-15);
    }

    #[test]
    fn test_second_derivative_coefficients() {
        let pairs: Vec<f64> = FDCoefficients::second_derivative_pairs(FdAccuracyOrder::Second);
        let center: f64 = FDCoefficients::second_derivative_center(FdAccuracyOrder::Second);

        assert_eq!(pairs.len(), 1);
        assert_relative_eq!(pairs[0], 1.0, epsilon = 1e-15);
        assert_relative_eq!(center, -2.0, epsilon = 1e-15);
    }

    #[test]
    fn test_generic_float_types() {
        let coeffs_f64: Vec<f64> = FDCoefficients::first_derivative(FdAccuracyOrder::Fourth);
        let coeffs_f32: Vec<f32> = FDCoefficients::first_derivative(FdAccuracyOrder::Fourth);

        assert_eq!(coeffs_f64.len(), coeffs_f32.len());
        for (c64, c32) in coeffs_f64.iter().zip(coeffs_f32.iter()) {
            assert_relative_eq!(*c64 as f32, *c32, epsilon = 1e-6);
        }
    }
}
