//! Apollo-backed FFT facade for kwavers.
//!
//! `kwavers` does not own a separate FFT engine. The canonical FFT plans,
//! caches, complex helpers, and real-to-complex transforms live in
//! `apollofft`; this module only reexports the Apollo API under the legacy
//! `kwavers::math::fft` path and keeps the spectral k-space utilities local.

pub mod gpu_fft;
pub mod kspace;
pub mod shift_operators;
pub mod utils;

pub use apollofft::application::cache::{
    get_fft_for_grid, Fft1dCache as FftCache1d, Fft1dCacheKey, Fft2dCache as FftCache2d,
    Fft2dCacheKey, Fft3dCache as FftCache3d, Fft3dCacheKey, FFT_CACHE, FFT_CACHE_1D, FFT_CACHE_2D,
    FFT_CACHE_3D,
};
pub use apollofft::{
    fft_1d_array, fft_1d_array_typed, fft_1d_complex, fft_1d_complex_inplace, fft_2d_array,
    fft_2d_array_typed, fft_2d_complex, fft_2d_complex_inplace, fft_3d_array, fft_3d_array_into,
    fft_3d_array_typed, fft_3d_complex, fft_3d_complex_inplace, fft_3d_complex_into, ifft_1d_array,
    ifft_1d_array_typed, ifft_1d_complex, ifft_1d_complex_inplace, ifft_2d_array,
    ifft_2d_array_typed, ifft_2d_complex, ifft_2d_complex_inplace, ifft_3d_array,
    ifft_3d_array_into, ifft_3d_array_typed, ifft_3d_complex, ifft_3d_complex_inplace, Complex32,
    Complex64, FftPlan1D as Fft1d, FftPlan2D as Fft2d, FftPlan3D as Fft3d, Normalization,
    ProcessorFft3d,
};

pub use gpu_fft::gpu_fft_available;
pub use kspace::KSpaceCalculator;
pub use utils::{analytic_signal_1d, apply_spectral_response_1d};
