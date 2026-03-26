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
