use num_traits::{Float, NumCast, Zero};

/// Generic numeric operations for improved type safety and reusability.
pub trait NumericOps<T>: Clone + Copy + PartialOrd + Zero
where
    T: Float + NumCast,
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
                .fold(T::zero(), |acc, val| acc + val),
        )
    }

    /// Generic vector normalization. Returns `false` if the vector has zero norm.
    fn normalize(vector: &mut [T]) -> bool {
        let norm_sq = vector
            .iter()
            .map(|&x| x * x)
            .fold(T::zero(), |acc, val| acc + val);
        if norm_sq <= T::zero() {
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
            .fold(T::zero(), |acc, val| acc + val)
            .sqrt()
    }

    /// Generic maximum absolute value.
    fn max_abs(array: &[T]) -> T {
        array
            .iter()
            .map(|&x| x.abs())
            .fold(T::zero(), |acc, val| acc.max(val))
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
