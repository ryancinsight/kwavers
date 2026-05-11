/// Machine epsilon for f64 (IEEE 754 double precision)
/// ε = 2^(-52) ≈ 2.220446049250313e-16
pub const F64_MACHINE_EPSILON: f64 = f64::EPSILON; // 2.220446049250313e-16

/// Unit roundoff for f64
/// u = 2^(-53) ≈ 1.1102230246251565e-16
pub const F64_UNIT_ROUNDOFF: f64 = F64_MACHINE_EPSILON / 2.0;

/// Default relative error tolerance for floating-point comparisons
/// Conservative threshold accounting for parallel reduction effects
/// Justification: ~4500× machine epsilon for n ≤ 10⁹
pub const DEFAULT_RELATIVE_TOLERANCE: f64 = 1e-12;

/// Default absolute error tolerance for near-zero values
/// Set well above subnormal threshold (≈5e-324) but below any physical signal
pub const DEFAULT_ABSOLUTE_TOLERANCE: f64 = 1e-15;

/// Number of warmup steps before equivalence measurement
/// Ensures solver has reached steady-state numerical behavior
pub const WARMUP_STEPS: usize = 5;

/// Number of measurement steps for equivalence validation
pub const MEASUREMENT_STEPS: usize = 100;

/// Maximum divergent points as fraction of total (0.0 for strict)
pub const MAX_DIVERGENT_FRACTION: f64 = 0.0;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test machine epsilon constant correctness
    /// IEEE 754 double precision: 2^(-52)
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_machine_epsilon_constant() {
        let expected = 2.0_f64.powi(-52);
        assert!((F64_MACHINE_EPSILON - expected).abs() < 1e-30);
        assert!((F64_MACHINE_EPSILON - 2.220446049250313e-16).abs() < 1e-30);
    }
}
