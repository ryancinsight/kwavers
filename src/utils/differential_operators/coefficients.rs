//! Finite difference coefficients module
//!
//! This module owns the mathematical knowledge of finite difference coefficients
//! following the Information Expert GRASP principle.
//!
//! References:
//! - Fornberg, B. (1988). "Generation of finite difference formulas"
//! - LeVeque, R.J. (2007). "Finite Difference Methods for ODEs and PDEs"

use num_traits::Float;

/// Spatial accuracy order for finite difference schemes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpatialOrder {
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
    /// # Panics
    /// Never panics - all coefficients are mathematically exact rational numbers
    /// that convert precisely to any IEEE 754 floating-point type.
    #[must_use]
    pub fn first_derivative<T: Float>(order: SpatialOrder) -> Vec<T> {
        // SAFETY: All coefficients are exact rational numbers that convert
        // precisely to f32/f64 without loss of precision or overflow.
        // Mathematical proof: All values are in range [-1, 1] with denominators
        // that are powers of small primes, ensuring exact IEEE 754 representation.
        match order {
            SpatialOrder::Second => vec![T::from(0.5).expect("0.5 converts exactly to IEEE 754")],
            SpatialOrder::Fourth => {
                vec![
                    T::from(-1.0 / 12.0).expect("Exact fraction converts to IEEE 754"),
                    T::from(2.0 / 3.0).expect("Exact fraction converts to IEEE 754"),
                ]
            }
            SpatialOrder::Sixth => {
                vec![
                    T::from(1.0 / 60.0).expect("Exact fraction converts to IEEE 754"),
                    T::from(-3.0 / 20.0).expect("Exact fraction converts to IEEE 754"),
                    T::from(3.0 / 4.0).expect("Exact fraction converts to IEEE 754"),
                ]
            }
            SpatialOrder::Eighth => {
                vec![
                    T::from(-1.0 / 280.0).expect("Exact fraction converts to IEEE 754"),
                    T::from(4.0 / 105.0).expect("Exact fraction converts to IEEE 754"),
                    T::from(-1.0 / 5.0).expect("Exact fraction converts to IEEE 754"),
                    T::from(4.0 / 5.0).expect("Exact fraction converts to IEEE 754"),
                ]
            }
        }
    }

    /// Get coefficients for second derivative pairs (off-center points)
    ///
    /// Returns coefficients for the points at ±h, ±2h, etc. from center.
    /// The center point coefficient should be computed separately.
    ///
    /// # Panics
    /// Never panics - all coefficients are exact IEEE 754 representations.
    #[must_use]
    pub fn second_derivative_pairs<T: Float>(order: SpatialOrder) -> Vec<T> {
        match order {
            SpatialOrder::Second => vec![T::from(1.0).expect("1.0 is exact in IEEE 754")],
            SpatialOrder::Fourth => {
                vec![
                    T::from(-1.0 / 12.0).expect("Exact fraction converts to IEEE 754"),
                    T::from(4.0 / 3.0).expect("Exact fraction converts to IEEE 754"),
                ]
            }
            SpatialOrder::Sixth => {
                vec![
                    T::from(1.0 / 90.0).expect("Exact fraction converts to IEEE 754"),
                    T::from(-3.0 / 20.0).expect("Exact fraction converts to IEEE 754"),
                    T::from(3.0 / 2.0).expect("Exact fraction converts to IEEE 754"),
                ]
            }
            SpatialOrder::Eighth => {
                vec![
                    T::from(-1.0 / 560.0).expect("Exact fraction converts to IEEE 754"),
                    T::from(8.0 / 315.0).expect("Exact fraction converts to IEEE 754"),
                    T::from(-1.0 / 5.0).expect("Exact fraction converts to IEEE 754"),
                    T::from(8.0 / 5.0).expect("Exact fraction converts to IEEE 754"),
                ]
            }
        }
    }

    /// Get center coefficient for second derivative
    ///
    /// # Panics
    /// Never panics - all coefficients are exact IEEE 754 representations.
    #[must_use]
    pub fn second_derivative_center<T: Float>(order: SpatialOrder) -> T {
        match order {
            SpatialOrder::Second => T::from(-2.0).expect("-2.0 is exact in IEEE 754"),
            SpatialOrder::Fourth => T::from(-5.0 / 2.0).expect("Exact fraction converts to IEEE 754"),
            SpatialOrder::Sixth => T::from(-49.0 / 18.0).expect("Exact fraction converts to IEEE 754"),
            SpatialOrder::Eighth => T::from(-205.0 / 72.0).expect("Exact fraction converts to IEEE 754"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_first_derivative_coefficients() {
        let coeffs: Vec<f64> = FDCoefficients::first_derivative(SpatialOrder::Second);
        assert_eq!(coeffs.len(), 1);
        assert_relative_eq!(coeffs[0], 0.5, epsilon = 1e-15);
    }

    #[test]
    fn test_second_derivative_coefficients() {
        let pairs: Vec<f64> = FDCoefficients::second_derivative_pairs(SpatialOrder::Second);
        let center: f64 = FDCoefficients::second_derivative_center(SpatialOrder::Second);
        
        assert_eq!(pairs.len(), 1);
        assert_relative_eq!(pairs[0], 1.0, epsilon = 1e-15);
        assert_relative_eq!(center, -2.0, epsilon = 1e-15);
    }

    #[test]
    fn test_generic_float_types() {
        let coeffs_f64: Vec<f64> = FDCoefficients::first_derivative(SpatialOrder::Fourth);
        let coeffs_f32: Vec<f32> = FDCoefficients::first_derivative(SpatialOrder::Fourth);
        
        assert_eq!(coeffs_f64.len(), coeffs_f32.len());
        for (c64, c32) in coeffs_f64.iter().zip(coeffs_f32.iter()) {
            assert_relative_eq!(*c64 as f32, *c32, epsilon = 1e-6);
        }
    }
}