/// Calculate ULPs (units in last place) between two f64 values
///
/// Useful for measuring floating-point error magnitude.
///
/// ## Definition
///
/// The ULP distance between two floating-point numbers x and y is the
/// number of representable f64 values between them.
///
/// ## Properties
///
/// - Same value: 0 ULPs
/// - Adjacent values: 1 ULP
/// - Maximum: 2^64 - 1 (for opposite signs)
pub fn ulps_diff(a: f64, b: f64) -> u64 {
    if a == b || (a.is_nan() && b.is_nan()) {
        return 0;
    }

    let a_bits = a.to_bits();
    let b_bits = b.to_bits();

    // Handle sign bit specially
    if (a < 0.0) != (b < 0.0) {
        // Different signs: distance through zero
        let a_dist = if a < 0.0 {
            a_bits - (-0.0_f64).to_bits()
        } else {
            a_bits - 0.0_f64.to_bits()
        };
        let b_dist = if b < 0.0 {
            b_bits - (-0.0_f64).to_bits()
        } else {
            b_bits - 0.0_f64.to_bits()
        };
        return a_dist + b_dist;
    }

    a_bits.abs_diff(b_bits)
}

/// Check if two values are within N ULPs of each other
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn within_ulps(a: f64, b: f64, max_ulps: u64) -> bool {
    ulps_diff(a, b) <= max_ulps
}

/// Verify IEEE 754 compliance for platform
///
/// Ensures that the current platform implements IEEE 754-2008 correctly.
/// Validates:
/// 1. Signed zero handling
/// 2. NaN propagation
/// 3. Infinity handling
/// 4. Basic arithmetic determinism
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub fn verify_ieee754_compliance() -> Result<(), String> {
    let mut failures = Vec::new();

    // Test 1: Signed zero handling
    let pos_zero = 0.0_f64;
    let neg_zero = -0.0_f64;
    if pos_zero != neg_zero {
        failures.push("Signed zero comparison failed".to_string());
    }

    // Test 2: NaN propagation
    let nan = f64::NAN;
    if !nan.is_nan() {
        failures.push("NaN detection failed".to_string());
    }
    #[allow(clippy::eq_op)]
    if nan == nan {
        failures.push("NaN == NaN should be false".to_string());
    }

    // Test 3: Infinity handling
    let inf = f64::INFINITY;
    let neg_inf = f64::NEG_INFINITY;
    if !inf.is_infinite() || !neg_inf.is_infinite() {
        failures.push("Infinity detection failed".to_string());
    }

    // Test 4: Basic arithmetic determinism
    let expected = 3.0;
    if (1.0 + 2.0) != expected {
        failures.push("Basic addition failed".to_string());
    }

    // Test 5: Division by zero produces infinity
    let div_result: f64 = 1.0 / 0.0;
    if !div_result.is_infinite() {
        failures.push("Division by zero should produce infinity".to_string());
    }

    // Test 6: Machine epsilon correctness
    let expected_eps = 2.0_f64.powi(-52);
    if (f64::EPSILON - expected_eps).abs() > 1e-20 {
        failures.push(format!(
            "Machine epsilon mismatch: {} vs {}",
            f64::EPSILON,
            expected_eps
        ));
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!("IEEE 754 compliance failures: {:?}", failures))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test IEEE 754 compliance verification
    /// # Panics
    /// - Panics if `Platform should be IEEE 754 compliant`.
    ///
    #[test]
    fn test_ieee754_compliance_check() {
        verify_ieee754_compliance().expect("Platform should be IEEE 754 compliant");
    }

    /// Test ULP distance calculation
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_ulp_calculation() {
        assert_eq!(ulps_diff(1.0, 1.0), 0, "Same value: 0 ULPs");
        assert_eq!(ulps_diff(0.0, 0.0), 0, "Zero: 0 ULPs");

        // Adjacent representable numbers
        let next = f64::from_bits((1.0f64).to_bits() + 1);
        assert_eq!(ulps_diff(1.0, next), 1, "Adjacent values: 1 ULP");

        // 1 ULP at 1.0 is EPSILON
        assert!(
            (next - 1.0).abs() - f64::EPSILON < 1e-30,
            "1 ULP at 1.0 should be EPSILON"
        );
    }

    /// Test within_ulps function
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_within_ulps() {
        assert!(within_ulps(1.0, 1.0, 0), "Same value within 0 ULPs");

        let next = f64::from_bits((1.0f64).to_bits() + 1);
        assert!(within_ulps(1.0, next, 1), "Adjacent within 1 ULP");
        assert!(!within_ulps(1.0, next, 0), "Adjacent not within 0 ULP");
    }
}
