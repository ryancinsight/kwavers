//! Apollo-backed FFT facade for kwavers.
//!
//! `kwavers` does not own a separate FFT engine. The canonical FFT plans,
//! caches, complex helpers, and real-to-complex transforms live in
//! `apollo`; this module only reexports the Apollo API under the legacy
//! `kwavers::math::fft` path and keeps the spectral k-space utilities local.

mod cache;
mod ext;
mod transforms;

pub mod gpu_fft;
pub mod kspace;
pub mod shift_operators;
pub mod spectral;

pub use apollo::{
    fftfreq, fftshift, ifftshift, rfftfreq, FftPlan1D, FftPlan2D, FftPlan3D, Normalization,
    PlanCacheProvider, Shape1D, Shape2D, Shape3D,
};
pub use cache::{
    get_fft_for_grid, Fft1d, Fft2d, Fft3d, FftCache1d, FftCache2d, FftCache3d, FFT_CACHE_1D,
    FFT_CACHE_2D, FFT_CACHE_3D,
};
pub use eunomia::{Complex32, Complex64};
pub use ext::{Fft2dInOutExt, Fft3dInOutExt};
pub use kspace::KSpaceCalculator;
pub use spectral::{analytic_signal_1d, apply_spectral_response_1d};
pub use transforms::{
    fft_1d_array, fft_1d_complex, fft_1d_complex_inplace, fft_1d_complex_slice_inplace,
    fft_2d_array, fft_2d_complex, fft_2d_complex_inplace, fft_3d_array, fft_3d_array_into,
    fft_3d_axis_complex_inplace, fft_3d_complex, fft_3d_complex_inplace, fft_3d_complex_into,
    ifft_1d_array, ifft_1d_complex, ifft_1d_complex_inplace, ifft_1d_complex_slice_inplace,
    ifft_2d_array, ifft_2d_complex, ifft_2d_complex_inplace, ifft_3d_array, ifft_3d_array_into,
    ifft_3d_axis_complex_inplace, ifft_3d_complex, ifft_3d_complex_inplace,
};
