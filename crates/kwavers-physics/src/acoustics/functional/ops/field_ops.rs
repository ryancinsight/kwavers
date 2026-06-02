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

impl<T: Send + Sync> FieldOps for Array3<T> {
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
        let (nx, ny, nz) = self.dim();
        let yz = ny * nz;

        // Derive coordinates from the flat index inside each Rayon task. This
        // removes the temporary `Vec<&T>` while preserving owned output order.
        let result: Vec<U> = (0..nx * ny * nz)
            .into_par_iter()
            .map(|flat| {
                let i = flat / yz;
                let rem = flat % yz;
                let j = rem / nz;
                let k = rem % nz;
                f(&self[[i, j, k]])
            })
            .collect();

        let shape = (nx, ny, nz);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Eq)]
    struct NonCloneValue(i32);

    fn small() -> Array3<f64> {
        Array3::from_shape_fn((2, 2, 2), |(i, j, k)| (i + j + k + 1) as f64)
    }

    /// map_field(×2): every element doubled compared to original.
    #[test]
    fn map_field_doubles_every_element() {
        let field = small();
        let doubled = field.map_field(|&x| x * 2.0);
        for ((i, j, k), &orig) in field.indexed_iter() {
            let expected = orig * 2.0;
            let got = doubled[[i, j, k]];
            assert!(
                (got - expected).abs() < 1e-14,
                "[{i},{j},{k}]: got {got}, expected {expected}"
            );
        }
    }

    /// fold_field sums all elements → equivalent to Array3::sum().
    #[test]
    fn fold_field_sums_all_elements() {
        let field = small();
        let expected: f64 = field.iter().sum();
        let got = field.fold_field(0.0, |acc, &x| acc + x);
        assert!(
            (got - expected).abs() < 1e-14,
            "fold sum: got {got}, expected {expected}"
        );
    }

    /// scan_field(0, +) produces running totals; last element equals total sum.
    #[test]
    fn scan_field_running_sum_last_equals_total_sum() {
        let field = small();
        let scan = field.scan_field(0.0, |&acc, &x| acc + x);
        let total: f64 = field.iter().sum();
        // Last element in row-major traversal accumulates all values
        let last = *scan.iter().last().unwrap();
        assert!(
            (last - total).abs() < 1e-14,
            "scan last={last}, total={total}"
        );
    }

    /// find_element(>2) returns the first element > 2 with its index.
    #[test]
    fn find_element_returns_first_matching_element() {
        let field = small();
        let result = field.find_element(|&x| x > 2.0);
        assert!(result.is_some(), "must find an element > 2");
        let (idx, &val) = result.unwrap();
        assert!(val > 2.0, "found value {val} at {idx:?} must be > 2");
    }

    /// count_matching(>1) counts elements satisfying predicate.
    ///
    /// In 2×2×2 with values 1..=3 (not all distinct), count(>1) = n cells with sum≥2.
    #[test]
    fn count_matching_counts_above_threshold() {
        let field = small();
        let count_above_1 = field.count_matching(|&x| x > 1.0);
        let expected = field.iter().filter(|&&x| x > 1.0).count();
        assert_eq!(
            count_above_1, expected,
            "count_matching > 1 must be {expected} (got {count_above_1})"
        );
        assert!(count_above_1 > 0, "must be at least one element > 1");
    }

    /// any(>10): false for field where max < 10; true when spike > 10 exists.
    #[test]
    fn any_returns_false_when_none_match_and_true_when_one_matches() {
        let field = small();
        assert!(!field.any(|&x| x > 10.0), "no element > 10 in small field");

        let mut spiked = field.clone();
        spiked[[1, 1, 1]] = 99.0;
        assert!(
            spiked.any(|&x| x > 10.0),
            "spike at [1,1,1] must trigger any()"
        );
    }

    /// all(>0): true for positive field; false when a zero element is present.
    #[test]
    fn all_returns_true_only_when_every_element_matches() {
        let field = small(); // all values ≥ 1
        assert!(field.all(|&x| x > 0.0), "all elements must be > 0");

        let mut with_zero = field.clone();
        with_zero[[0, 0, 0]] = 0.0;
        assert!(
            !with_zero.all(|&x| x > 0.0),
            "zero element must break all()"
        );
    }

    /// Read-only field operations must not require cloneable elements.
    #[test]
    fn read_only_ops_accept_non_clone_elements() {
        let field = Array3::from_shape_fn((2, 2, 2), |(i, j, k)| NonCloneValue((i + j + k) as i32));

        let mapped = field.map_field(|value| value.0 * 3);
        let par_mapped = field.par_map_field(|value| value.0 * 3);
        let folded = field.fold_field(0, |acc, value| acc + value.0);
        let filtered: Vec<_> = field.filter_indices(|value| value.0 == 2).collect();

        assert_eq!(mapped[[1, 1, 1]], 9);
        assert_eq!(par_mapped[[1, 1, 1]], 9);
        assert_eq!(folded, 12);
        assert_eq!(filtered, vec![(0, 1, 1), (1, 0, 1), (1, 1, 0)]);
    }
}
