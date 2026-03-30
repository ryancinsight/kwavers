// fft/mod.rs
pub mod fft_processor;
pub mod kspace;
pub mod shift_operators;
pub mod utils;

pub use fft_processor::{Fft1d, Fft2d, Fft3d};
pub use kspace::KSpaceCalculator;
pub type ProcessorFft3d = Fft3d;

use ndarray::{Array1, Array2, Array3};
pub use num_complex::Complex64;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

pub fn fft_1d_array(field: &Array1<f64>) -> Array1<Complex64> {
    let n = field.len();
    FFT_CACHE_1D.get_or_create(n).forward(field)
}

pub fn fft_2d_array(field: &Array2<f64>) -> Array2<Complex64> {
    let (nx, ny) = field.dim();
    FFT_CACHE_2D.get_or_create(nx, ny).forward(field)
}

pub fn fft_3d_array(field: &Array3<f64>) -> Array3<Complex64> {
    let (nx, ny, nz) = field.dim();
    FFT_CACHE_3D.get_or_create(nx, ny, nz).forward(field)
}

pub fn ifft_1d_array(field_hat: &Array1<Complex64>) -> Array1<f64> {
    let n = field_hat.len();
    FFT_CACHE_1D.get_or_create(n).inverse(field_hat)
}

pub fn ifft_2d_array(field_hat: &Array2<Complex64>) -> Array2<f64> {
    let (nx, ny) = field_hat.dim();
    FFT_CACHE_2D.get_or_create(nx, ny).inverse(field_hat)
}

pub fn ifft_3d_array(field_hat: &Array3<Complex64>) -> Array3<f64> {
    let (nx, ny, nz) = field_hat.dim();
    FFT_CACHE_3D.get_or_create(nx, ny, nz).inverse(field_hat)
}

/// Forward 1D FFT of complex input, operating on a mutable reference.
///
/// This is the zero-copy path. The array is transformed in place and the
/// modified reference is returned for chaining. Prefer this over
/// [`fft_1d_complex`] whenever the caller does not need to retain the original.
pub fn fft_1d_complex_inplace(data: &mut Array1<Complex64>) {
    let n = data.len();
    FFT_CACHE_1D.get_or_create(n).forward_complex_inplace(data);
}

/// Inverse 1D FFT of complex input in place, including 1/N normalisation.
pub fn ifft_1d_complex_inplace(data: &mut Array1<Complex64>) {
    let n = data.len();
    FFT_CACHE_1D
        .get_or_create(n)
        .inverse_complex_inplace(data);
    let norm = 1.0 / n as f64;
    data.mapv_inplace(|c| c * norm);
}

/// Forward 1D FFT of complex input, returning a new array.
///
/// Allocates one copy of `field`. Prefer [`fft_1d_complex_inplace`] when the
/// original array is no longer needed after the call.
pub fn fft_1d_complex(field: &Array1<Complex64>) -> Array1<Complex64> {
    let mut data = field.to_owned();
    fft_1d_complex_inplace(&mut data);
    data
}

/// Inverse 1D FFT of complex input, returning a new array with 1/N normalisation.
pub fn ifft_1d_complex(field_hat: &Array1<Complex64>) -> Array1<Complex64> {
    let mut data = field_hat.to_owned();
    ifft_1d_complex_inplace(&mut data);
    data
}

/// Forward 2D FFT of complex input in place.
pub fn fft_2d_complex_inplace(data: &mut Array2<Complex64>) {
    let (nx, ny) = data.dim();
    FFT_CACHE_2D
        .get_or_create(nx, ny)
        .forward_complex_inplace(data);
}

/// Inverse 2D FFT of complex input in place, including 1/(NxNy) normalisation.
pub fn ifft_2d_complex_inplace(data: &mut Array2<Complex64>) {
    let (nx, ny) = data.dim();
    FFT_CACHE_2D
        .get_or_create(nx, ny)
        .inverse_complex_inplace(data);
    let norm = 1.0 / (nx * ny) as f64;
    data.mapv_inplace(|c| c * norm);
}

/// Forward 2D FFT of complex input, returning a new array.
pub fn fft_2d_complex(field: &Array2<Complex64>) -> Array2<Complex64> {
    let mut data = field.to_owned();
    fft_2d_complex_inplace(&mut data);
    data
}

/// Inverse 2D FFT of complex input, returning a new array with 1/(NxNy) normalisation.
pub fn ifft_2d_complex(field_hat: &Array2<Complex64>) -> Array2<Complex64> {
    let mut data = field_hat.to_owned();
    ifft_2d_complex_inplace(&mut data);
    data
}

/// Forward 3D FFT of complex input in place.
///
/// Zero allocation — the cached [`Fft3d`] plan is used directly on the
/// provided buffer.
pub fn fft_3d_complex_inplace(data: &mut Array3<Complex64>) {
    let (nx, ny, nz) = data.dim();
    FFT_CACHE_3D
        .get_or_create(nx, ny, nz)
        .forward_complex_inplace(data);
}

/// Inverse 3D FFT of complex input in place, including 1/(NxNyNz) normalisation.
pub fn ifft_3d_complex_inplace(data: &mut Array3<Complex64>) {
    let (nx, ny, nz) = data.dim();
    FFT_CACHE_3D
        .get_or_create(nx, ny, nz)
        .inverse_complex_inplace(data);
    let norm = 1.0 / (nx * ny * nz) as f64;
    data.mapv_inplace(|c| c * norm);
}

/// Forward 3D FFT of complex input, returning a new array.
///
/// Allocates exactly one copy of `field`. Prefer [`fft_3d_complex_inplace`]
/// when the original is not needed after the call.
pub fn fft_3d_complex(field: &Array3<Complex64>) -> Array3<Complex64> {
    let mut data = field.to_owned();
    fft_3d_complex_inplace(&mut data);
    data
}

/// Inverse 3D FFT of complex input, returning a new array with 1/(NxNyNz) normalisation.
pub fn ifft_3d_complex(field_hat: &Array3<Complex64>) -> Array3<Complex64> {
    let mut data = field_hat.to_owned();
    ifft_3d_complex_inplace(&mut data);
    data
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fft1dCacheKey(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fft2dCacheKey(usize, usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fft3dCacheKey {
    nx: usize,
    ny: usize,
    nz: usize,
}

pub struct FftCache1d {
    cache: RwLock<HashMap<Fft1dCacheKey, Arc<Fft1d>>>,
}

pub struct FftCache2d {
    cache: RwLock<HashMap<Fft2dCacheKey, Arc<Fft2d>>>,
}

pub struct FftCache3d {
    cache: RwLock<HashMap<Fft3dCacheKey, Arc<Fft3d>>>,
}

// Debug implementations for FFT cache structures
impl std::fmt::Debug for FftCache1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cache = self.cache.read();
        f.debug_struct("FftCache1d")
            .field("cache_size", &cache.len())
            .finish()
    }
}

impl std::fmt::Debug for FftCache2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cache = self.cache.read();
        f.debug_struct("FftCache2d")
            .field("cache_size", &cache.len())
            .finish()
    }
}

impl std::fmt::Debug for FftCache3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cache = self.cache.read();
        f.debug_struct("FftCache3d")
            .field("cache_size", &cache.len())
            .finish()
    }
}

impl FftCache1d {
    pub fn get_or_create(&self, n: usize) -> Arc<Fft1d> {
        let key = Fft1dCacheKey(n);
        {
            let cache = self.cache.read();
            if let Some(fft) = cache.get(&key) {
                return Arc::clone(fft);
            }
        }
        let mut cache = self.cache.write();
        if let Some(fft) = cache.get(&key) {
            return Arc::clone(fft);
        }
        let fft = Arc::new(Fft1d::new(n));
        cache.insert(key, Arc::clone(&fft));
        fft
    }
}

impl FftCache2d {
    pub fn get_or_create(&self, nx: usize, ny: usize) -> Arc<Fft2d> {
        let key = Fft2dCacheKey(nx, ny);
        {
            let cache = self.cache.read();
            if let Some(fft) = cache.get(&key) {
                return Arc::clone(fft);
            }
        }
        let mut cache = self.cache.write();
        if let Some(fft) = cache.get(&key) {
            return Arc::clone(fft);
        }
        let fft = Arc::new(Fft2d::new(nx, ny));
        cache.insert(key, Arc::clone(&fft));
        fft
    }
}

impl FftCache3d {
    pub fn get_or_create(&self, nx: usize, ny: usize, nz: usize) -> Arc<Fft3d> {
        let key = Fft3dCacheKey { nx, ny, nz };
        {
            let cache = self.cache.read();
            if let Some(fft) = cache.get(&key) {
                return Arc::clone(fft);
            }
        }
        let mut cache = self.cache.write();
        if let Some(fft) = cache.get(&key) {
            return Arc::clone(fft);
        }
        let fft = Arc::new(Fft3d::new(nx, ny, nz));
        cache.insert(key, Arc::clone(&fft));
        fft
    }
}

pub static FFT_CACHE_1D: Lazy<FftCache1d> = Lazy::new(|| FftCache1d {
    cache: RwLock::new(HashMap::new()),
});
pub static FFT_CACHE_2D: Lazy<FftCache2d> = Lazy::new(|| FftCache2d {
    cache: RwLock::new(HashMap::new()),
});
pub static FFT_CACHE_3D: Lazy<FftCache3d> = Lazy::new(|| FftCache3d {
    cache: RwLock::new(HashMap::new()),
});

// Compatibility re-export
pub static FFT_CACHE: Lazy<FftCache3d> = Lazy::new(|| FftCache3d {
    cache: RwLock::new(HashMap::new()),
});

pub fn get_fft_for_grid(nx: usize, ny: usize, nz: usize) -> Arc<Fft3d> {
    FFT_CACHE_3D.get_or_create(nx, ny, nz)
}
