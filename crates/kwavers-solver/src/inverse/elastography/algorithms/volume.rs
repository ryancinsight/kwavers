//! Internal mutable 3-D scalar-volume abstraction for elastography filters.

use leto::Array3 as LetoArray3;
use ndarray::Array3 as NdArray3;

/// Mutable 3-D `f64` volume used by elastography smoothing and boundary fills.
pub trait Volume3 {
    /// Return the volume shape as `(nx, ny, nz)`.
    fn dimensions(&self) -> (usize, usize, usize);

    /// Return the value at `index`.
    fn value(&self, index: [usize; 3]) -> f64;

    /// Set the value at `index`.
    fn set_value(&mut self, index: [usize; 3], value: f64);
}

impl Volume3 for NdArray3<f64> {
    fn dimensions(&self) -> (usize, usize, usize) {
        self.dim()
    }

    fn value(&self, [i, j, k]: [usize; 3]) -> f64 {
        self[[i, j, k]]
    }

    fn set_value(&mut self, [i, j, k]: [usize; 3], value: f64) {
        self[[i, j, k]] = value;
    }
}

impl Volume3 for LetoArray3<f64> {
    fn dimensions(&self) -> (usize, usize, usize) {
        let [nx, ny, nz] = self.shape();
        (nx, ny, nz)
    }

    fn value(&self, [i, j, k]: [usize; 3]) -> f64 {
        self[[i, j, k]]
    }

    fn set_value(&mut self, [i, j, k]: [usize; 3], value: f64) {
        self[[i, j, k]] = value;
    }
}
