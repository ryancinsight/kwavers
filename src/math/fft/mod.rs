// fft/mod.rs
pub mod fft_processor;
pub mod kspace;
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

pub fn fft_1d_complex(field: &Array1<Complex64>) -> Array1<Complex64> {
    let n = field.len();
    let mut data = field.clone();
    FFT_CACHE_1D
        .get_or_create(n)
        .forward_complex_inplace(&mut data);
    data
}

pub fn ifft_1d_complex(field_hat: &Array1<Complex64>) -> Array1<Complex64> {
    let n = field_hat.len();
    let mut data = field_hat.clone();
    FFT_CACHE_1D
        .get_or_create(n)
        .inverse_complex_inplace(&mut data);
    let norm = 1.0 / n as f64;
    data.mapv_inplace(|c| c * norm);
    data
}

pub fn fft_2d_complex(field: &Array2<Complex64>) -> Array2<Complex64> {
    let (nx, ny) = field.dim();
    let mut data = field.clone();
    FFT_CACHE_2D
        .get_or_create(nx, ny)
        .forward_complex_inplace(&mut data);
    data
}

pub fn ifft_2d_complex(field_hat: &Array2<Complex64>) -> Array2<Complex64> {
    let (nx, ny) = field_hat.dim();
    let mut data = field_hat.clone();
    FFT_CACHE_2D
        .get_or_create(nx, ny)
        .inverse_complex_inplace(&mut data);
    let norm = 1.0 / (nx * ny) as f64;
    data.mapv_inplace(|c| c * norm);
    data
}

pub fn fft_3d_complex(field: &Array3<Complex64>) -> Array3<Complex64> {
    let (nx, ny, nz) = field.dim();
    FFT_CACHE_3D
        .get_or_create(nx, ny, nz)
        .forward_complex(field)
}

pub fn ifft_3d_complex(field_hat: &Array3<Complex64>) -> Array3<Complex64> {
    let (nx, ny, nz) = field_hat.dim();
    let mut data = FFT_CACHE_3D
        .get_or_create(nx, ny, nz)
        .inverse_complex(field_hat);
    let norm = 1.0 / (nx * ny * nz) as f64;
    data.mapv_inplace(|c| c * norm);
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
