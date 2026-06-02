//! Extension trait that adds iterator methods to `dyn Medium`.

use crate::grid::Grid;
use crate::medium::Medium;

use super::interface::InterfaceIterator;
use super::parallel::ParallelMediumIterator;
use super::property::MediumPropertyIterator;

/// Extension trait to add iterator methods to Medium
pub trait MediumIteratorExt {
    /// Create an iterator over medium properties
    fn iter_properties<'a>(&'a self, grid: &'a Grid) -> MediumPropertyIterator<'a>;

    /// Create an interface iterator
    fn iter_interfaces<'a>(&'a self, grid: &'a Grid, threshold: f64) -> InterfaceIterator<'a>;

    /// Create a parallel iterator
    fn par_iter<'a>(&'a self, grid: &'a Grid) -> ParallelMediumIterator<'a>;
}

impl MediumIteratorExt for dyn Medium {
    fn iter_properties<'a>(&'a self, grid: &'a Grid) -> MediumPropertyIterator<'a> {
        MediumPropertyIterator::new(self, grid)
    }

    fn iter_interfaces<'a>(&'a self, grid: &'a Grid, threshold: f64) -> InterfaceIterator<'a> {
        InterfaceIterator::new(self, grid, threshold)
    }

    fn par_iter<'a>(&'a self, grid: &'a Grid) -> ParallelMediumIterator<'a> {
        ParallelMediumIterator::new(self, grid)
    }
}
