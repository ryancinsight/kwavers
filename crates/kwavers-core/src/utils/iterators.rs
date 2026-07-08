// src/utils/iterators.rs
//! Zero-cost iterator abstractions for efficient physics simulations.
//!
//! Uses ndarray view types to maintain compatibility with callers in kwavers-solver
//! and other partially-migrated crates. Will be updated to leto views once all
//! callers are migrated.

use moirai_parallel::{enumerate_mut_with, for_each_index_with, for_each_mut_with, Adaptive};
use ndarray::{ArrayBase, ArrayView3, ArrayViewMut3, DataMut, Dimension};

/// Apply an indexed mutation over a 3-D ndarray view.
pub fn for_each_indexed_mut<T, F>(mut values: ArrayViewMut3<'_, T>, f: F)
where
    T: Send,
    F: Fn((usize, usize, usize), &mut T) + Send + Sync,
{
    let (nx, ny, nz) = values.dim();
    if let Some(slice) = values.as_slice_mut() {
        let f_ref = &f;
        enumerate_mut_with::<Adaptive, _, _>(slice, |idx, value| {
            let plane = ny * nz;
            let i = idx / plane;
            let rem = idx % plane;
            f_ref((i, rem / nz, rem % nz), value);
        });
    } else {
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    f((i, j, k), &mut values[[i, j, k]]);
                }
            }
        }
    }
}

/// Apply an indexed mutation over paired 3-D ndarray views.
pub fn for_each_indexed_pair_mut<T, U, F>(
    mut values: ArrayViewMut3<'_, T>,
    input: ArrayView3<'_, U>,
    f: F,
) where
    T: Send,
    U: Sync,
    F: Fn((usize, usize, usize), &mut T, &U) + Send + Sync,
{
    debug_assert_eq!(
        values.dim(),
        input.dim(),
        "invariant: paired 3-D traversal shape mismatch"
    );

    let (_nx, ny, nz) = values.dim();
    match (values.as_slice_mut(), input.as_slice()) {
        (Some(values_slice), Some(input_slice)) => {
            let f_ref = &f;
            enumerate_mut_with::<Adaptive, _, _>(values_slice, |idx, value| {
                let plane = ny * nz;
                let i = idx / plane;
                let rem = idx % plane;
                f_ref((i, rem / nz, rem % nz), value, &input_slice[idx]);
            });
        }
        _ => {
            let (nx, ny, nz) = values.dim();
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        f((i, j, k), &mut values[[i, j, k]], &input[[i, j, k]]);
                    }
                }
            }
        }
    }
}

/// Apply a scalar value transform to an ndarray in place.
pub fn apply_inplace<T, S, D, F>(values: &mut ArrayBase<S, D>, f: F)
where
    T: Copy + Send,
    S: DataMut<Elem = T>,
    D: Dimension,
    F: Fn(T) -> T + Send + Sync,
{
    if let Some(slice) = values.as_slice_memory_order_mut() {
        for_each_mut_with::<Adaptive, _, _>(slice, |value| {
            *value = f(*value);
        });
    } else {
        values.map_inplace(|value| {
            *value = f(*value);
        });
    }
}

/// Iterator for processing 3D grid points with spatial coordinates.
pub struct GridPointIterator<'a, T> {
    array: ArrayViewMut3<'a, T>,
    nx: usize,
    ny: usize,
    nz: usize,
}

impl<'a, T> GridPointIterator<'a, T>
where
    T: Send + Sync + Clone + Default,
{
    #[must_use]
    pub fn new(array: ArrayViewMut3<'a, T>) -> Self {
        let (nx, ny, nz) = array.dim();
        Self { array, nx, ny, nz }
    }

    #[must_use]
    pub fn get(&self, i: usize, j: usize, k: usize) -> Option<&T> {
        if i < self.nx && j < self.ny && k < self.nz {
            Some(&self.array[[i, j, k]])
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, i: usize, j: usize, k: usize) -> Option<&mut T> {
        if i < self.nx && j < self.ny && k < self.nz {
            Some(&mut self.array[[i, j, k]])
        } else {
            None
        }
    }

    pub fn process_interior_sequential<F>(&mut self, dx: f64, dy: f64, dz: f64, mut processor: F)
    where
        F: FnMut(usize, usize, usize, f64, f64, f64, &mut T),
    {
        for i in 1..self.nx - 1 {
            for j in 1..self.ny - 1 {
                for k in 1..self.nz - 1 {
                    let x = i as f64 * dx;
                    let y = j as f64 * dy;
                    let z = k as f64 * dz;
                    processor(i, j, k, x, y, z, &mut self.array[[i, j, k]]);
                }
            }
        }
    }
}

/// Iterator for chunked processing with cache-friendly access patterns.
pub struct ChunkedProcessor<'a, T> {
    array: ArrayViewMut3<'a, T>,
    nx: usize,
    ny: usize,
    nz: usize,
}

impl<'a, T> ChunkedProcessor<'a, T>
where
    T: Sync + Send,
{
    #[must_use]
    pub fn new(array: ArrayViewMut3<'a, T>) -> Self {
        let (nx, ny, nz) = array.dim();
        Self { array, nx, ny, nz }
    }

    pub fn process_interior<F>(&mut self, dx: f64, dy: f64, dz: f64, processor: F)
    where
        F: Fn(usize, usize, usize, f64, f64, f64) + Sync + Send,
    {
        for i in 1..self.nx - 1 {
            for j in 1..self.ny - 1 {
                for k in 1..self.nz - 1 {
                    let x = i as f64 * dx;
                    let y = j as f64 * dy;
                    let z = k as f64 * dz;
                    processor(i, j, k, x, y, z);
                }
            }
        }
    }

    pub fn get_mut(&mut self, i: usize, j: usize, k: usize) -> Option<&mut T> {
        if i < self.nx && j < self.ny && k < self.nz {
            Some(&mut self.array[[i, j, k]])
        } else {
            None
        }
    }
}

/// Gradient computer using central differences.
pub struct IteratorGradientComputer<'a> {
    array: ArrayView3<'a, f64>,
    nx: usize,
    ny: usize,
    nz: usize,
}

impl<'a> IteratorGradientComputer<'a> {
    #[must_use]
    pub fn new(array: ArrayView3<'a, f64>) -> Self {
        let (nx, ny, nz) = array.dim();
        Self { array, nx, ny, nz }
    }

    pub fn compute_interior_gradients<F>(&self, dx: f64, dy: f64, dz: f64, processor: F)
    where
        F: Fn(f64, f64, f64, usize, usize, usize) + Sync + Send,
    {
        let dx_inv = 1.0 / (2.0 * dx);
        let dy_inv = 1.0 / (2.0 * dy);
        let dz_inv = 1.0 / (2.0 * dz);

        let interior_x = self.nx.saturating_sub(2);
        for_each_index_with::<Adaptive, _>(interior_x, |offset| {
            let i = offset + 1;
            for j in 1..self.ny - 1 {
                for k in 1..self.nz - 1 {
                    let gx = (self.array[[i + 1, j, k]] - self.array[[i - 1, j, k]]) * dx_inv;
                    let gy = (self.array[[i, j + 1, k]] - self.array[[i, j - 1, k]]) * dy_inv;
                    let gz = (self.array[[i, j, k + 1]] - self.array[[i, j, k - 1]]) * dz_inv;
                    processor(gx, gy, gz, i, j, k);
                }
            }
        });
    }

    #[must_use]
    pub fn collect_gradients(&self, dx: f64, dy: f64, dz: f64)
        -> (ndarray::Array3<f64>, ndarray::Array3<f64>, ndarray::Array3<f64>)
    {
        let mut gx = ndarray::Array3::zeros((self.nx, self.ny, self.nz));
        let mut gy = ndarray::Array3::zeros((self.nx, self.ny, self.nz));
        let mut gz = ndarray::Array3::zeros((self.nx, self.ny, self.nz));

        let dx_inv = 1.0 / (2.0 * dx);
        let dy_inv = 1.0 / (2.0 * dy);
        let dz_inv = 1.0 / (2.0 * dz);

        for i in 1..self.nx - 1 {
            for j in 1..self.ny - 1 {
                for k in 1..self.nz - 1 {
                    gx[[i, j, k]] =
                        (self.array[[i + 1, j, k]] - self.array[[i - 1, j, k]]) * dx_inv;
                    gy[[i, j, k]] =
                        (self.array[[i, j + 1, k]] - self.array[[i, j - 1, k]]) * dy_inv;
                    gz[[i, j, k]] =
                        (self.array[[i, j, k + 1]] - self.array[[i, j, k - 1]]) * dz_inv;
                }
            }
        }

        (gx, gy, gz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn apply_inplace_updates_values() {
        let mut data = Array3::from_shape_vec((1, 2, 3), vec![1i32, 2, 3, 4, 5, 6])
            .expect("shape matches data length");
        apply_inplace(&mut data, |value| value * value);
        let values: Vec<i32> = data.iter().copied().collect();
        assert_eq!(values, vec![1, 4, 9, 16, 25, 36]);
    }

    #[test]
    fn test_chunked_processor() {
        let mut data = Array3::<f64>::zeros((10, 10, 10));
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    data[[i, j, k]] = (i * 100 + j * 10 + k) as f64;
                }
            }
        }
        let mut proc = ChunkedProcessor::new(data.view_mut());
        assert_eq!(proc.get_mut(1, 2, 3), Some(&mut 123.0));
        proc.process_interior(1.0, 1.0, 1.0, |i, j, k, x, y, z| {
            assert_eq!(x, i as f64);
            assert_eq!(y, j as f64);
            assert_eq!(z, k as f64);
        });
    }

    #[test]
    fn test_gradient_computer() {
        let mut array = Array3::<f64>::zeros((10, 10, 10));
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    array[[i, j, k]] = (i + j + k) as f64;
                }
            }
        }
        let computer = IteratorGradientComputer::new(array.view());
        let count = std::sync::atomic::AtomicUsize::new(0);
        computer.compute_interior_gradients(1.0, 1.0, 1.0, |gx, gy, gz, _, _, _| {
            assert!((gx - 1.0).abs() < 1e-10);
            assert!((gy - 1.0).abs() < 1e-10);
            assert!((gz - 1.0).abs() < 1e-10);
            count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 512);
    }
}