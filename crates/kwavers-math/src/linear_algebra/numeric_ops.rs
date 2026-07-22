//! Atlas-migrated (Phase-1A pilot): legacy NumTraits float bounds replaced by
//! `eunomia::RealField` (re-exported from `eunomia::traits::field`) and the
//! `NumericElement::ZERO` associated constant.
//!
//! Atlas replacement map (this file only; the rest of kwavers-math remains on
//! legacy until the Phase-1B sweep — see kwavers-math `Cargo.toml` comment):
//!
//! | num_traits       | eunomia                            |
//! |------------------|-----------------------------------|
//! | `Float`          | `eunomia::RealField`              |
//! | `NumCast`        | (vestigial in this file; dropped) |
//! | `<_>::zero()`   | `T::ZERO` (`<T as NumericElement>`)|
//!
//! Source of truth: <https://eunomia.dev/traits/real_field>
//!
//! # Design Rationale
//!
//! `NumericOps<T>` is the generic numeric primitive layer for kwavers-*. It
//! supplies reusable reduce / element-wise / safe-divide helpers to every
//! downstream kwavers Math consumer. Porting it to eunomia's scalar SSOT
//! removes the last legacy float-trait boundary in this file.

use eunomia::RealField;

/// Generic numeric operations for improved type safety and reusability.
pub trait NumericOps<T>: Copy + PartialOrd
where
    T: RealField,
{
    /// Generic dot product for any float type.
    fn dot_product(a: &[T], b: &[T]) -> Option<T> {
        if a.len() != b.len() {
            return None;
        }
        Some(
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| x * y)
                .fold(T::ZERO, |acc, val| acc + val),
        )
    }

    /// Generic vector normalization. Returns `false` if the vector has zero norm.
    fn normalize(vector: &mut [T]) -> bool {
        let norm_sq = vector
            .iter()
            .map(|&x| x * x)
            .fold(T::ZERO, |acc, val| acc + val);
        if norm_sq <= T::ZERO {
            return false;
        }
        let norm = norm_sq.sqrt();
        for x in vector.iter_mut() {
            *x = *x / norm;
        }
        true
    }

    /// Generic element-wise addition for arrays.
    fn add_arrays(a: &[T], b: &[T], out: &mut [T]) -> Result<(), &'static str> {
        if a.len() != b.len() || b.len() != out.len() {
            return Err("Array length mismatch");
        }
        for ((a_val, b_val), out_val) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
            *out_val = *a_val + *b_val;
        }
        Ok(())
    }

    /// Generic scalar multiplication.
    fn scale_array(input: &[T], scalar: T, out: &mut [T]) -> Result<(), &'static str> {
        if input.len() != out.len() {
            return Err("Array length mismatch");
        }
        for (input_val, out_val) in input.iter().zip(out.iter_mut()) {
            *out_val = *input_val * scalar;
        }
        Ok(())
    }

    /// Generic L2 norm calculation.
    fn l2_norm(array: &[T]) -> T {
        array
            .iter()
            .map(|&x| x * x)
            .fold(T::ZERO, |acc, val| acc + val)
            .sqrt()
    }

    /// Generic maximum absolute value.
    fn max_abs(array: &[T]) -> T {
        array
            .iter()
            .map(|&x| x.abs())
            .fold(T::ZERO, |acc, val| if val > acc { val } else { acc })
    }

    /// Safe division with tolerance check.
    fn safe_divide(numerator: T, denominator: T, tolerance: T) -> Option<T> {
        if denominator.abs() > tolerance {
            Some(numerator / denominator)
        } else {
            None
        }
    }
}

impl NumericOps<Self> for f64 {}
impl NumericOps<Self> for f32 {}

#[cfg(test)]
mod tests {
    //! Numerical regression tests pinning `NumericOps::dot_product`,
    //! `l2_norm`, and `max_abs` to closed-form float answers.
    //!
    //! Drift surfaces as a numerical failure (not a silent compile regression)
    //! when a future change touches:
    //! - `<T as eunomia::NumericElement>::ZERO` initial value (kwavers-math
    //!   uses `T::ZERO` directly in the fold identity per the Phase-1A port);
    //! - the `NumericElement` operator supertrait chain (`+`, `*`, `abs`, `sqrt`);
    //! - the `PartialOrd`-driven fold in `max_abs` (eunomia::RealField
    //!   deliberately does not propagate a `max` method, so the fold is
    //!   `if val > acc { val } else { acc }`).
    //!
    //! Comparison strategy:
    //! - `assert_eq!` for integer-summed / integer-squared-then-sqrt cases —
    //!   bit-exact under IEEE 754, the strongest possible detector.
    //! - `eunomia::assert_relative_eq!` (epsilon 1e-12) only for the
    //!   transcendental case `l2_norm([1, 2]) = sqrt(5)`.
    //!
    //! # Test scope (Phase-1A regression — not exhaustive)
    //!
    //! Pinned: dot_product, l2_norm, max_abs, plus f32 parity subset.
    //! Not pinned: safe_divide, add_arrays, scale_array, normalize.

    use super::NumericOps;
    use eunomia::assert_relative_eq;

    // ───── dot_product ───────────────────────────────────────────

    #[test]
    fn dot_product_basic_f64() {
        // 1*4 + 2*5 + 3*6 = 32
        let a = [1.0_f64, 2.0, 3.0];
        let b = [4.0_f64, 5.0, 6.0];
        assert_eq!(f64::dot_product(&a, &b), Some(32.0));
    }

    #[test]
    fn dot_product_mixed_signs_f64() {
        // 1*4 + (-2)*5 + 3*6 = 4 - 10 + 18 = 12
        let a = [1.0_f64, -2.0, 3.0];
        let b = [4.0_f64, 5.0, 6.0];
        assert_eq!(f64::dot_product(&a, &b), Some(12.0));
    }

    #[test]
    fn dot_product_zero_initialised_lhs_f64() {
        // Pinned against `<T as eunomia::NumericElement>::ZERO`: the fold
        // identity must be a real zero, not a sentinel.
        let a = [0.0_f64, 0.0, 0.0];
        let b = [99.0_f64, 99.0, 99.0];
        assert_eq!(f64::dot_product(&a, &b), Some(0.0));
    }

    #[test]
    fn dot_product_empty_slices_f64() {
        // Fold over an empty iterator returns the identity `T::ZERO`.
        let a: [f64; 0] = [];
        let b: [f64; 0] = [];
        assert_eq!(f64::dot_product(&a, &b), Some(0.0));
    }

    #[test]
    fn dot_product_mismatched_lengths_returns_none_f64() {
        let a = [1.0_f64, 2.0];
        let b = [3.0_f64];
        assert_eq!(f64::dot_product(&a, &b), None);
    }

    // ───── l2_norm ─────────────────────────────────────────────

    #[test]
    fn l2_norm_classic_3_4_5_f64() {
        // 3² + 4² = 25; sqrt(25) = 5 — bit-exact under IEEE 754.
        let v = [3.0_f64, 4.0];
        assert_eq!(f64::l2_norm(&v), 5.0);
    }

    #[test]
    fn l2_norm_negative_components_f64() {
        // Squaring removes the sign, so the answer is identical to the
        // positive-input case (operator-chain check).
        let v = [-3.0_f64, -4.0];
        assert_eq!(f64::l2_norm(&v), 5.0);
    }

    #[test]
    fn l2_norm_zero_array_f64() {
        // Pinned against `<T as eunomia::NumericElement>::ZERO`: a fold of
        // zeros must be ZERO before `.sqrt()`. If `ZERO` ever drifts to a
        // sentinel (NaN-sentinel pattern?), this fails loudly.
        let v = [0.0_f64, 0.0, 0.0, 0.0];
        assert_eq!(f64::l2_norm(&v), 0.0);
    }

    #[test]
    fn l2_norm_empty_f64() {
        // Empty-fold yields `T::ZERO`; sqrt(0) = 0.
        let v: [f64; 0] = [];
        assert_eq!(f64::l2_norm(&v), 0.0);
    }

    #[test]
    fn l2_norm_transcendental_case_f64() {
        // 1² + 2² = 5; sqrt(5) is irrational — the only non-bit-exact case.
        // Epsilon 1e-12 catches real drift, not 64-bit round-off.
        let v = [1.0_f64, 2.0];
        assert_relative_eq!(f64::l2_norm(&v), 2.23606797749979_f64, epsilon = 1e-12);
        // sqrt(5) exact to ~16 digits
    }

    // ───── max_abs ────────────────────────────────────────────────

    #[test]
    fn max_abs_mixed_signs_f64() {
        // 10 max wins against -7 / 3 / 5.
        let v = [-7.0_f64, 3.0, -10.0, 5.0];
        assert_eq!(f64::max_abs(&v), 10.0);
    }

    #[test]
    fn max_abs_all_positive_f64() {
        let v = [1.0_f64, 2.0, 3.0, 4.0];
        assert_eq!(f64::max_abs(&v), 4.0);
    }

    #[test]
    fn max_abs_all_negative_f64() {
        let v = [-1.0_f64, -2.0, -3.0];
        assert_eq!(f64::max_abs(&v), 3.0);
    }

    #[test]
    fn max_abs_zero_array_f64() {
        // Pinned against `<T as eunomia::NumericElement>::ZERO`: the fold
        // identity is ZERO, so a zero array returns ZERO via the
        // `PartialOrd`-driven fold.
        let v = [0.0_f64, 0.0, 0.0];
        assert_eq!(f64::max_abs(&v), 0.0);
    }

    #[test]
    fn max_abs_fractional_inputs_f64() {
        let v = [0.5_f64, 0.25, 0.125];
        assert_eq!(f64::max_abs(&v), 0.5);
    }

    #[test]
    fn max_abs_empty_f64() {
        // Empty-fold yields `T::ZERO`; the PartialOrd-driven fold returns
        // ZERO (no iteration occurs).
        let v: [f64; 0] = [];
        assert_eq!(f64::max_abs(&v), 0.0);
    }

    // ───── f32 monomorphization parity ───────────────────────────────────────
    // Single representative subset to catch `<f32 as eunomia::RealField>::ZERO`
    // and the f32 path's operator-chain wiring without bloating the suite.

    #[test]
    fn dot_product_f32_parity() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        assert_eq!(f32::dot_product(&a, &b), Some(32.0_f32));
    }

    #[test]
    fn l2_norm_f32_classic_3_4_5() {
        let v = [3.0_f32, 4.0];
        assert_eq!(f32::l2_norm(&v), 5.0_f32);
    }

    #[test]
    fn max_abs_f32_mixed_signs() {
        let v = [-7.0_f32, 3.0, -10.0, 5.0];
        assert_eq!(f32::max_abs(&v), 10.0_f32);
    }
}
