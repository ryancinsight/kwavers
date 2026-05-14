//! Apollo-backed FFT facade for kwavers.
//!
//! `kwavers` does not own a separate FFT engine. The canonical FFT plans,
//! caches, complex helpers, and real-to-complex transforms live in
//! `apollo`; this module only reexports the Apollo API under the legacy
//! `kwavers::math::fft` path and keeps the spectral k-space utilities local.

pub mod gpu_fft;
pub mod kspace;
pub mod shift_operators;
pub mod utils;

pub use apollo::{
    fft_1d_array, fft_1d_array_typed, fft_1d_complex, fft_1d_complex_inplace, fft_2d_array,
    fft_2d_array_typed, fft_2d_complex, fft_2d_complex_inplace, fft_3d_array, fft_3d_array_into,
    fft_3d_array_typed, fft_3d_complex, fft_3d_complex_inplace, fft_3d_complex_into, fft_3d_r2c,
    fft_3d_r2c_into, ifft_1d_array, ifft_1d_array_typed, ifft_1d_complex, ifft_1d_complex_inplace,
    ifft_2d_array, ifft_2d_array_typed, ifft_2d_complex, ifft_2d_complex_inplace, ifft_3d_array,
    ifft_3d_array_into, ifft_3d_array_typed, ifft_3d_complex, ifft_3d_complex_inplace, ifft_3d_r2c,
    ifft_3d_r2c_into, Complex32, Complex64, FftPlan1D as Fft1d, FftPlan2D as Fft2d,
    FftPlan3D as Fft3d, Normalization, Shape1D, Shape2D, Shape3D,
};
pub use apollo::{
    fftfreq, fftshift, get_fft_for_grid, ifftshift, rfftfreq, Fft1dCache as FftCache1d,
    Fft1dCacheKey, Fft2dCache as FftCache2d, Fft2dCacheKey, Fft3dCache as FftCache3d,
    Fft3dCacheKey, FFT_CACHE_1D, FFT_CACHE_2D, FFT_CACHE_3D,
};

/// Compatibility alias for the legacy `apollo::FFT_CACHE` static (now `FFT_CACHE_3D`).
///
/// The apollo 0.11 refactor split the unified `FFT_CACHE` global into per-dimension
/// caches `FFT_CACHE_1D`, `FFT_CACHE_2D`, and `FFT_CACHE_3D`. All historical
/// kwavers call sites used the 3-D variant, so this alias preserves the prior
/// import surface without weakening the contract.
pub use apollo::FFT_CACHE_3D as FFT_CACHE;

/// Compatibility alias for the legacy `apollo::ProcessorFft3d` type.
///
/// The apollo 0.11 refactor consolidated the `ProcessorFft3d` 3-D processor
/// into the unified `FftPlan3D` plan type. Both share the same `forward` /
/// `inverse` real/complex contract and the same `Shape3D` constructor input,
/// so this alias preserves the prior kwavers import surface.
pub use apollo::FftPlan3D as ProcessorFft3d;

pub use gpu_fft::gpu_fft_available;
pub use kspace::KSpaceCalculator;
pub use utils::{analytic_signal_1d, apply_spectral_response_1d};

use ndarray::{Array2, Array3, Zip};

/// Full-spectrum (nx, ny, nz) complex-to-complex 3-D transforms with caller-owned
/// real and complex storage.
///
/// The apollo 0.11 refactor removed `FftPlan{2D,3D}::forward_into` and
/// `inverse_into` in favor of the explicit `forward_r2c_into` /
/// `inverse_c2r_into` (half-spectrum r2c/c2r) and `forward_complex_inplace` /
/// `inverse_complex_inplace` (full-spectrum c2c) splits. Every kwavers call
/// site that used the old `forward_into` / `inverse_into` was full-spectrum
/// c2c with `Array3<Complex64>::zeros((nx, ny, nz))` outputs, so this
/// extension trait re-binds the previous surface to the new c2c inplace
/// methods. The transform is mathematically identical; the migration is
/// purely an API rename driven by the apollo split.
pub trait Fft3dInOutExt {
    /// Forward 3-D FFT of a real field into a caller-owned full-spectrum
    /// complex buffer. Equivalent to assigning `field + 0i` into `out` and
    /// running an in-place complex forward FFT.
    fn forward_into(&self, field: &Array3<f64>, out: &mut Array3<Complex64>);

    /// Inverse 3-D FFT of a full-spectrum complex field into a caller-owned
    /// real buffer using a caller-owned complex scratch. Equivalent to
    /// copying `field_hat` into `scratch`, running an in-place complex
    /// inverse FFT on `scratch`, and assigning the real component into `out`.
    fn inverse_into(
        &self,
        field_hat: &Array3<Complex64>,
        out: &mut Array3<f64>,
        scratch: &mut Array3<Complex64>,
    );
}

/// 2-D counterpart to [`Fft3dInOutExt`] with identical semantics.
pub trait Fft2dInOutExt {
    /// Forward 2-D FFT of a real field into a caller-owned full-spectrum
    /// complex buffer.
    fn forward_into(&self, field: &Array2<f64>, out: &mut Array2<Complex64>);

    /// Inverse 2-D FFT of a full-spectrum complex field into a caller-owned
    /// real buffer using a caller-owned complex scratch.
    fn inverse_into(
        &self,
        field_hat: &Array2<Complex64>,
        out: &mut Array2<f64>,
        scratch: &mut Array2<Complex64>,
    );
}

impl Fft2dInOutExt for Fft2d {
    #[inline]
    fn forward_into(&self, field: &Array2<f64>, out: &mut Array2<Complex64>) {
        debug_assert_eq!(
            field.dim(),
            out.dim(),
            "Fft2dInOutExt::forward_into: shape mismatch between real input and complex output"
        );
        Zip::from(out.view_mut())
            .and(field)
            .par_for_each(|dst, &src| {
                *dst = Complex64::new(src, 0.0);
            });
        self.forward_complex_inplace(out);
    }

    #[inline]
    fn inverse_into(
        &self,
        field_hat: &Array2<Complex64>,
        out: &mut Array2<f64>,
        scratch: &mut Array2<Complex64>,
    ) {
        debug_assert_eq!(
            field_hat.dim(),
            scratch.dim(),
            "Fft2dInOutExt::inverse_into: shape mismatch between complex input and complex scratch"
        );
        debug_assert_eq!(
            field_hat.dim(),
            out.dim(),
            "Fft2dInOutExt::inverse_into: shape mismatch between complex input and real output"
        );
        scratch.assign(field_hat);
        self.inverse_complex_inplace(scratch);
        Zip::from(out.view_mut())
            .and(scratch.view())
            .par_for_each(|dst, src| *dst = src.re);
    }
}

impl Fft3dInOutExt for Fft3d {
    #[inline]
    fn forward_into(&self, field: &Array3<f64>, out: &mut Array3<Complex64>) {
        debug_assert_eq!(
            field.dim(),
            out.dim(),
            "Fft3dInOutExt::forward_into: shape mismatch between real input and complex output"
        );
        Zip::from(out.view_mut())
            .and(field)
            .par_for_each(|dst, &src| {
                *dst = Complex64::new(src, 0.0);
            });
        self.forward_complex_inplace(out);
    }

    #[inline]
    fn inverse_into(
        &self,
        field_hat: &Array3<Complex64>,
        out: &mut Array3<f64>,
        scratch: &mut Array3<Complex64>,
    ) {
        debug_assert_eq!(
            field_hat.dim(),
            scratch.dim(),
            "Fft3dInOutExt::inverse_into: shape mismatch between complex input and complex scratch"
        );
        debug_assert_eq!(
            field_hat.dim(),
            out.dim(),
            "Fft3dInOutExt::inverse_into: shape mismatch between complex input and real output"
        );
        scratch.assign(field_hat);
        self.inverse_complex_inplace(scratch);
        Zip::from(out.view_mut())
            .and(scratch.view())
            .par_for_each(|dst, src| *dst = src.re);
    }
}
