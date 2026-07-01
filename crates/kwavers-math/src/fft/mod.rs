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
    fft_3d_array_typed, fft_3d_complex, fft_3d_complex_inplace, fft_3d_complex_into, fftfreq,
    fftshift, ifft_1d_array, ifft_1d_array_typed, ifft_1d_complex, ifft_1d_complex_inplace,
    ifft_2d_array, ifft_2d_array_typed, ifft_2d_complex, ifft_2d_complex_inplace, ifft_3d_array,
    ifft_3d_array_into, ifft_3d_array_typed, ifft_3d_complex, ifft_3d_complex_inplace, ifftshift,
    rfftfreq, Complex32, Complex64, FftPlan1D, FftPlan2D, FftPlan3D, Normalization,
    PlanCacheProvider, Shape1D, Shape2D, Shape3D,
};
use std::sync::Arc;

/// f64-specialized FFT plan aliases. Apollo's `FftPlan{N}D` became generic over
/// `F: MixedRadixScalar`; the kwavers spectral layer is f64 throughout, so this
/// anti-corruption boundary binds `F = f64` and exposes a single concrete plan
/// type to the rest of the codebase (PSTD derivatives, KZK, k-space correction).
pub type Fft1d = FftPlan1D<f64>;
/// 2-D f64 FFT plan. See [`Fft1d`].
pub type Fft2d = FftPlan2D<f64>;
/// 3-D f64 FFT plan. See [`Fft1d`].
pub type Fft3d = FftPlan3D<f64>;

/// Cached 3-D FFT plan for a grid shape. Anti-corruption adapter over apollo's
/// `PlanCacheProvider` trait (which replaced the removed global `get_fft_for_grid`
/// free function). Resolves to the per-scalar thread-local + global plan cache that
/// `f64::get_3d_plan` maintains, so repeated calls at one shape reuse the plan.
#[inline]
#[must_use]
pub fn get_fft_for_grid(nx: usize, ny: usize, nz: usize) -> Arc<Fft3d> {
    <f64 as PlanCacheProvider>::get_3d_plan(Shape3D { nx, ny, nz })
}

/// Zero-sized facade preserving the `FFT_CACHE_1D.get_or_create(shape)` call surface.
/// Apollo replaced its global `FFT_CACHE_*` statics with the precision-keyed
/// [`PlanCacheProvider`] trait; this re-binds the legacy call sites to
/// `f64::get_1d_plan` without changing any caller.
#[derive(Debug, Clone, Copy, Default)]
pub struct FftCache1d;
impl FftCache1d {
    /// Retrieve or instantiate the cached 1-D plan for `shape`.
    #[inline]
    #[must_use]
    pub fn get_or_create(&self, shape: Shape1D) -> Arc<Fft1d> {
        <f64 as PlanCacheProvider>::get_1d_plan(shape)
    }
}

/// 2-D counterpart to [`FftCache1d`].
#[derive(Debug, Clone, Copy, Default)]
pub struct FftCache2d;
impl FftCache2d {
    /// Retrieve or instantiate the cached 2-D plan for `shape`.
    #[inline]
    #[must_use]
    pub fn get_or_create(&self, shape: Shape2D) -> Arc<Fft2d> {
        <f64 as PlanCacheProvider>::get_2d_plan(shape)
    }
}

/// 3-D counterpart to [`FftCache1d`].
#[derive(Debug, Clone, Copy, Default)]
pub struct FftCache3d;
impl FftCache3d {
    /// Retrieve or instantiate the cached 3-D plan for `shape`.
    #[inline]
    #[must_use]
    pub fn get_or_create(&self, shape: Shape3D) -> Arc<Fft3d> {
        <f64 as PlanCacheProvider>::get_3d_plan(shape)
    }
}

/// Process-wide cached 1-D FFT plan provider. See [`FftCache1d`].
pub static FFT_CACHE_1D: FftCache1d = FftCache1d;
/// Process-wide cached 2-D FFT plan provider. See [`FftCache2d`].
pub static FFT_CACHE_2D: FftCache2d = FftCache2d;
/// Process-wide cached 3-D FFT plan provider. See [`FftCache3d`].
pub static FFT_CACHE_3D: FftCache3d = FftCache3d;

pub use gpu_fft::gpu_fft_available;
pub use kspace::KSpaceCalculator;
pub use utils::{analytic_signal_1d, apply_spectral_response_1d};

use ndarray::{s, Array2, Array3, Zip};
use std::cell::RefCell;

thread_local! {
    /// Per-thread full-spectrum `(nx, ny, nz)` complex scratch used by the
    /// half-spectrum r2c/c2r emulation in [`Fft3dInOutExt`]. Apollo dropped its
    /// public half-spectrum transforms; this scratch lets the ACL run apollo's
    /// full-spectrum complex plan and truncate/expand to the `nz_c = nz/2 + 1`
    /// layout the PSTD core still uses. Resized on grid-shape change, then reused
    /// across timesteps (zero steady-state allocation for a fixed grid).
    static R2C_FULL_SCRATCH: RefCell<Array3<Complex64>> = RefCell::new(Array3::zeros((0, 0, 0)));

    /// Per-thread half-spectrum `(nx, ny, nz_c)` complex scratch for the c2r
    /// transposed-axis inverse, kept separate from the caller's scratch argument
    /// (which some callers reuse as an input across calls).
    static R2C_HALF_SCRATCH: RefCell<Array3<Complex64>> = RefCell::new(Array3::zeros((0, 0, 0)));

    /// Per-thread `(nx, ny, nz/2)` packed-real working buffer for the even-nz
    /// forward r2c z-pass (`FftPlan3D::forward_real_z_into`).
    static R2C_PACK_SCRATCH: RefCell<Array3<Complex64>> = RefCell::new(Array3::zeros((0, 0, 0)));
}

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

    /// Forward real-to-complex 3-D FFT writing the **half-spectrum** `(nx, ny,
    /// nz/2+1)` of a real field. The contiguous z-axis is transformed to the
    /// half-spectrum — for even `nz` via apollo's **packed-real** transform
    /// (`forward_real_z_into`: a length-`nz/2` complex FFT + Hermitian unpack, half
    /// the z-FFT work and z scratch), for odd `nz` via a full c2c z-pass then
    /// truncation — after which the transposed y and x axes are transformed on the
    /// **half** with apollo's batched tiled per-axis FFT. The y/x passes are
    /// per-z-slice independent, so the result is bit-identical to a full c2c
    /// followed by truncation. `half_out` must have shape `(nx, ny, nz/2+1)`.
    fn forward_r2c_into(&self, real: &Array3<f64>, half_out: &mut Array3<Complex64>);

    /// Inverse complex-to-real 3-D FFT from a **half-spectrum** `(nx, ny,
    /// nz/2+1)` into a real field. Inverse-transforms the transposed x and y axes
    /// on the half-spectrum first (per-z-slice independent), reconstructs the full
    /// z-spectrum — after the x/y inverse the upper z-slices are the plain
    /// conjugate z-mirror of the lower (the `(nx-i, ny-j)` reflection is absorbed
    /// by `IDFT(conj(X[-n])) = conj(IDFT(X))`) — then inverses along z. Bit-
    /// identical to a full Hermitian reconstruction + full c2c inverse, at ~half
    /// the transposed x/y work. The `scratch` argument is retained for call-site
    /// compatibility and is unused (a thread-local half scratch is used instead).
    /// `half_in` must have shape `(nx, ny, nz/2+1)`.
    fn inverse_c2r_into(
        &self,
        half_in: &Array3<Complex64>,
        out: &mut Array3<f64>,
        scratch: &mut Array3<Complex64>,
    );

    /// Forward full-spectrum 3-D FFT of a real field, allocating the output.
    fn forward(&self, real: &Array3<f64>) -> Array3<Complex64>;

    /// Inverse full-spectrum 3-D FFT to a real field, allocating the output.
    fn inverse(&self, spectrum: &Array3<Complex64>) -> Array3<f64>;
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

    #[inline]
    fn forward_r2c_into(&self, real: &Array3<f64>, half_out: &mut Array3<Complex64>) {
        let (nx, ny, nz) = real.dim();
        let nz_c = nz / 2 + 1;
        debug_assert_eq!(
            half_out.dim(),
            (nx, ny, nz_c),
            "forward_r2c_into: half_out must be (nx, ny, nz/2+1)"
        );
        // Transform the contiguous z-axis to the half-spectrum, then the
        // transposed y/x axes on that half (per-z-slice independent ⇒ bit-
        // identical to a full c2c + truncation, at ~half the y/x work). For even
        // nz the z-pass is a **packed-real** transform (length-nz/2 complex FFT +
        // Hermitian unpack), halving the z-FFT work and z scratch; odd nz falls
        // back to a full c2c z-pass on the thread-local full scratch.
        if nz % 2 == 0 {
            R2C_PACK_SCRATCH.with(|cell| {
                let mut borrow = cell.borrow_mut();
                let m = nz / 2;
                if borrow.dim() != (nx, ny, m) {
                    *borrow = Array3::<Complex64>::zeros((nx, ny, m));
                }
                self.forward_real_z_into(real, half_out, &mut borrow);
            });
        } else {
            R2C_FULL_SCRATCH.with(|cell| {
                let mut borrow = cell.borrow_mut();
                if borrow.dim() != (nx, ny, nz) {
                    *borrow = Array3::<Complex64>::zeros((nx, ny, nz));
                }
                let full: &mut Array3<Complex64> = &mut borrow;
                Zip::from(full.view_mut())
                    .and(real)
                    .par_for_each(|dst, &src| *dst = Complex64::new(src, 0.0));
                self.forward_axis_complex_inplace(full, 2);
                half_out.assign(&full.slice(s![.., .., 0..nz_c]));
            });
        }
        let half_plan = get_fft_for_grid(nx, ny, nz_c);
        half_plan.forward_axis_complex_inplace(half_out, 1);
        half_plan.forward_axis_complex_inplace(half_out, 0);
    }

    #[inline]
    fn inverse_c2r_into(
        &self,
        half_in: &Array3<Complex64>,
        out: &mut Array3<f64>,
        _scratch: &mut Array3<Complex64>,
    ) {
        let (nx, ny, nz) = out.dim();
        let nz_c = nz / 2 + 1;
        debug_assert_eq!(
            half_in.dim(),
            (nx, ny, nz_c),
            "inverse_c2r_into: half_in must be (nx, ny, nz/2+1)"
        );
        // Inverse along the transposed x and y axes on the half-spectrum first
        // (per-z-slice independent, on a thread-local half scratch so the caller's
        // `_scratch` — which some callers reuse as a later input — is untouched),
        // then reconstruct the full z-spectrum and inverse along z. After the x/y
        // inverse the upper z-slices are the plain conjugate z-mirror of the lower
        // — the (nx-i, ny-j) reflection of the Hermitian symmetry is absorbed by
        // the identity `IDFT(conj(X[-n])) = conj(IDFT(X))` — so this is bit-
        // identical to a full Hermitian reconstruction followed by a full c2c
        // inverse, at ~half the tiled-transpose x/y work.
        R2C_HALF_SCRATCH.with(|hcell| {
            let mut hborrow = hcell.borrow_mut();
            if hborrow.dim() != (nx, ny, nz_c) {
                *hborrow = Array3::<Complex64>::zeros((nx, ny, nz_c));
            }
            let half: &mut Array3<Complex64> = &mut hborrow;
            half.assign(half_in);
            let half_plan = get_fft_for_grid(nx, ny, nz_c);
            half_plan.inverse_axis_complex_inplace(half, 0);
            half_plan.inverse_axis_complex_inplace(half, 1);
            R2C_FULL_SCRATCH.with(|cell| {
                let mut borrow = cell.borrow_mut();
                if borrow.dim() != (nx, ny, nz) {
                    *borrow = Array3::<Complex64>::zeros((nx, ny, nz));
                }
                let full: &mut Array3<Complex64> = &mut borrow;
                full.slice_mut(s![.., .., 0..nz_c]).assign(half);
                for k in nz_c..nz {
                    let kk = nz - k;
                    for i in 0..nx {
                        for j in 0..ny {
                            full[[i, j, k]] = half[[i, j, kk]].conj();
                        }
                    }
                }
                self.inverse_axis_complex_inplace(full, 2);
                Zip::from(out.view_mut())
                    .and(full.view())
                    .par_for_each(|dst, src| *dst = src.re);
            });
        });
    }

    #[inline]
    fn forward(&self, real: &Array3<f64>) -> Array3<Complex64> {
        let mut out = real.mapv(|v| Complex64::new(v, 0.0));
        self.forward_complex_inplace(&mut out);
        out
    }

    #[inline]
    fn inverse(&self, spectrum: &Array3<Complex64>) -> Array3<f64> {
        let mut tmp = spectrum.clone();
        self.inverse_complex_inplace(&mut tmp);
        tmp.mapv(|c| c.re)
    }
}

#[cfg(test)]
mod r2c_optimized_tests {
    use super::{get_fft_for_grid, Fft3dInOutExt};
    use ndarray::{s, Array3};
    use num_complex::Complex64;

    fn check_shape(nx: usize, ny: usize, nz: usize) {
        let nz_c = nz / 2 + 1;
        let fft = get_fft_for_grid(nx, ny, nz);
        let real = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            let x = ((i * 131 + j * 17 + k * 7) % 101) as f64 / 101.0 - 0.5;
            (x * 6.283).sin() + 0.3 * x + 0.1
        });

        // (1) Optimized forward_r2c is bit-identical to a full c2c + truncation.
        let mut half_new = Array3::zeros((nx, ny, nz_c));
        fft.forward_r2c_into(&real, &mut half_new);
        let mut full = real.mapv(|v| Complex64::new(v, 0.0));
        fft.forward_complex_inplace(&mut full);
        let ref_half = full.slice(s![.., .., 0..nz_c]).to_owned();
        let fwd_err = half_new
            .iter()
            .zip(ref_half.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(
            fwd_err < 1e-9,
            "forward_r2c({nx},{ny},{nz}) vs full-c2c reference: {fwd_err:.2e}"
        );

        // (2) Optimized inverse_c2r recovers the real field (round-trip).
        let mut out = Array3::zeros((nx, ny, nz));
        let mut scratch = Array3::zeros((nx, ny, nz_c));
        fft.inverse_c2r_into(&half_new, &mut out, &mut scratch);
        let rt_err = out
            .iter()
            .zip(real.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            rt_err < 1e-9,
            "r2c→c2r round-trip ({nx},{ny},{nz}): {rt_err:.2e}"
        );
    }

    #[test]
    fn optimized_r2c_c2r_matches_reference_and_roundtrips() {
        check_shape(8, 6, 10); // even nz
        check_shape(7, 5, 9); // odd nz
        check_shape(16, 16, 16); // power-of-two cube
        check_shape(12, 1, 8); // degenerate y (2-D-like)
    }
}
