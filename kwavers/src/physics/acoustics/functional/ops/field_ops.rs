//! Field Operations trait with iterator support

use ndarray::Array3;
use rayon::prelude::*;

/// Field operations trait with iterator support
pub trait FieldOps {
    type Item;

    /// Map a function over all elements
    fn map_field<F, U>(&self, f: F) -> Array3<U>
    where
        F: FnMut(&Self::Item) -> U;

    /// Filter and return iterator of matching indices (lazy evaluation)
    fn filter_indices<'a, F>(
        &'a self,
        predicate: F,
    ) -> impl Iterator<Item = (usize, usize, usize)> + 'a
    where
        F: Fn(&Self::Item) -> bool + 'a;

    /// Fold over the field with an accumulator
    fn fold_field<F, U>(&self, init: U, f: F) -> U
    where
        F: Fn(U, &Self::Item) -> U;

    /// Scan over the field, producing intermediate results
    fn scan_field<F, U>(&self, init: U, f: F) -> Array3<U>
    where
        F: Fn(&U, &Self::Item) -> U,
        U: Clone;

    /// Parallel map operation for better performance
    fn par_map_field<F, U>(&self, f: F) -> Array3<U>
    where
        F: Fn(&Self::Item) -> U + Sync + Send,
        U: Send + Sync,
        Self::Item: Sync;

    /// Find the first element matching a predicate
    fn find_element<F>(&self, predicate: F) -> Option<((usize, usize, usize), &Self::Item)>
    where
        F: Fn(&Self::Item) -> bool;

    /// Count elements matching a predicate
    fn count_matching<F>(&self, predicate: F) -> usize
    where
        F: Fn(&Self::Item) -> bool;

    /// Check if any element matches a predicate
    fn any<F>(&self, predicate: F) -> bool
    where
        F: Fn(&Self::Item) -> bool;

    /// Check if all elements match a predicate
    fn all<F>(&self, predicate: F) -> bool
    where
        F: Fn(&Self::Item) -> bool;
}

impl<T: Clone + Send + Sync> FieldOps for Array3<T> {
    type Item = T;

    fn map_field<F, U>(&self, mut f: F) -> Array3<U>
    where
        F: FnMut(&Self::Item) -> U,
    {
        let shape = self.dim();
        Array3::from_shape_fn(shape, |(i, j, k)| f(&self[[i, j, k]]))
    }

    fn filter_indices<'a, F>(
        &'a self,
        predicate: F,
    ) -> impl Iterator<Item = (usize, usize, usize)> + 'a
    where
        F: Fn(&Self::Item) -> bool + 'a,
    {
        self.indexed_iter()
            .filter(move |(_, val)| predicate(val))
            .map(|(idx, _)| idx)
    }

    fn fold_field<F, U>(&self, init: U, f: F) -> U
    where
        F: Fn(U, &Self::Item) -> U,
    {
        self.iter().fold(init, f)
    }

    fn scan_field<F, U>(&self, init: U, f: F) -> Array3<U>
    where
        F: Fn(&U, &Self::Item) -> U,
        U: Clone,
    {
        let shape = self.dim();
        let mut result = Array3::from_elem(shape, init.clone());
        let mut accumulator = init;

        for ((i, j, k), val) in self.indexed_iter() {
            accumulator = f(&accumulator, val);
            result[[i, j, k]] = accumulator.clone();
        }

        result
    }

    fn par_map_field<F, U>(&self, f: F) -> Array3<U>
    where
        F: Fn(&Self::Item) -> U + Sync + Send,
        U: Send + Sync,
        Self::Item: Sync,
    {
        let shape = self.dim();
        let result: Vec<U> = self
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(f)
            .collect();

        Array3::from_shape_vec(shape, result).expect("Shape mismatch in parallel map operation")
    }

    fn find_element<F>(&self, predicate: F) -> Option<((usize, usize, usize), &Self::Item)>
    where
        F: Fn(&Self::Item) -> bool,
    {
        self.indexed_iter().find(|(_, val)| predicate(val))
    }

    fn count_matching<F>(&self, predicate: F) -> usize
    where
        F: Fn(&Self::Item) -> bool,
    {
        self.iter().filter(|&val| predicate(val)).count()
    }

    fn any<F>(&self, predicate: F) -> bool
    where
        F: Fn(&Self::Item) -> bool,
    {
        self.iter().any(predicate)
    }

    fn all<F>(&self, predicate: F) -> bool
    where
        F: Fn(&Self::Item) -> bool,
    {
        self.iter().all(predicate)
    }
}
