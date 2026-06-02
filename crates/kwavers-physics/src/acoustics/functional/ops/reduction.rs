//! Reduction operations over fields

use ndarray::Array3;

/// Reduction operations over fields
pub trait FieldReduction<T> {
    /// Compute the sum of all elements
    fn sum(&self) -> T
    where
        T: std::iter::Sum;

    /// Compute the product of all elements
    fn product(&self) -> T
    where
        T: std::iter::Product;

    /// Find the minimum element
    fn min(&self) -> Option<&T>
    where
        T: Ord;

    /// Find the maximum element
    fn max(&self) -> Option<&T>
    where
        T: Ord;

    /// Compute the mean (average) of all elements
    fn mean(&self) -> T
    where
        T: std::iter::Sum + std::ops::Div<usize, Output = T> + Clone;
}

impl<T: Clone> FieldReduction<T> for Array3<T> {
    fn sum(&self) -> T
    where
        T: std::iter::Sum,
    {
        self.iter().cloned().sum()
    }

    fn product(&self) -> T
    where
        T: std::iter::Product,
    {
        self.iter().cloned().product()
    }

    fn min(&self) -> Option<&T>
    where
        T: Ord,
    {
        self.iter().min()
    }

    fn max(&self) -> Option<&T>
    where
        T: Ord,
    {
        self.iter().max()
    }

    fn mean(&self) -> T
    where
        T: std::iter::Sum + std::ops::Div<usize, Output = T> + Clone,
    {
        let sum: T = self.iter().cloned().sum();
        sum / self.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_by_two() -> Array3<i32> {
        // Values 1..=8 in row-major order
        Array3::from_shape_fn((2, 2, 2), |(i, j, k)| (i * 4 + j * 2 + k + 1) as i32)
    }

    /// product() = Π elements. For 1·2·3·4·5·6·7·8 = 40320.
    #[test]
    fn product_equals_factorial_8_for_sequential_elements() {
        let field = two_by_two();
        let p = field.product();
        assert_eq!(
            p, 40320_i32,
            "product of 1..=8 must be 8! = 40320 (got {p})"
        );
    }

    /// min() returns reference to the smallest element (1 for 1..=8).
    #[test]
    fn min_returns_smallest_element() {
        let field = two_by_two();
        let m = field.min().copied().unwrap();
        assert_eq!(m, 1, "min must be 1 (got {m})");
    }

    /// max() returns reference to the largest element (8 for 1..=8).
    #[test]
    fn max_returns_largest_element() {
        let field = two_by_two();
        let m = field.max().copied().unwrap();
        assert_eq!(m, 8, "max must be 8 (got {m})");
    }

    /// min() on a uniform array returns the uniform value.
    #[test]
    fn min_uniform_field_returns_uniform_value() {
        let field: Array3<i32> = Array3::from_elem((3, 3, 3), 42);
        assert_eq!(field.min().copied().unwrap(), 42);
    }

    /// max() on a uniform array returns the uniform value.
    #[test]
    fn max_uniform_field_returns_uniform_value() {
        let field: Array3<i32> = Array3::from_elem((3, 3, 3), 7);
        assert_eq!(field.max().copied().unwrap(), 7);
    }
}
