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
    fftfreq, fftshift, ifftshift, rfftfreq, FftPlan1D, FftPlan2D, FftPlan3D, Normalization,
    PlanCacheProvider, Shape1D, Shape2D, Shape3D,
};
pub use eunomia::{Complex32, Complex64};
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

pub use kspace::KSpaceCalculator;
pub use utils::{analytic_signal_1d, apply_spectral_response_1d};

use leto::{Array1, Array2, Array3};
use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};
use std::cell::RefCell;

const FFT_ASSIGN_CHUNK_LEN: usize = 4096;

thread_local! {
    /// Per-thread full-spectrum `(nx, ny, nz)` complex scratch used by the
    /// half-spectrum r2c/c2r emulation in [`Fft3dInOutExt`]. Apollo dropped its
    /// public half-spectrum transforms; this scratch lets the ACL run apollo's
    /// full-spectrum complex plan and truncate/expand to the `nz_c = nz/2 + 1`
    /// layout the PSTD core still uses. Resized on grid-shape change, then reused
    /// across timesteps (zero steady-state allocation for a fixed grid).
    static R2C_FULL_SCRATCH: RefCell<Array3<Complex64>> =
        RefCell::new(Array3::from_elem([0, 0, 0], Complex64::default()));
}

/// Forward 1-D FFT of a real Leto array through Apollo's Leto-owned engine.
#[must_use]
pub fn fft_1d_array(field: &Array1<f64>) -> Array1<Complex64> {
    apollo::fft_1d_array(field)
}

/// Inverse 1-D FFT of a complex Leto array, returning the real component.
#[must_use]
pub fn ifft_1d_array(field_hat: &Array1<Complex64>) -> Array1<f64> {
    apollo::ifft_1d_array(field_hat)
}

/// Forward 1-D complex FFT, allocating output.
#[must_use]
pub fn fft_1d_complex(field: &Array1<Complex64>) -> Array1<Complex64> {
    let mut out = field.clone();
    fft_1d_complex_inplace(&mut out);
    out
}

/// Inverse 1-D complex FFT, allocating output.
#[must_use]
pub fn ifft_1d_complex(field_hat: &Array1<Complex64>) -> Array1<Complex64> {
    let mut out = field_hat.clone();
    ifft_1d_complex_inplace(&mut out);
    out
}

/// Forward 1-D complex FFT in place.
pub fn fft_1d_complex_inplace(data: &mut Array1<Complex64>) {
    apollo::fft_1d_complex_inplace(data);
}

/// Inverse 1-D complex FFT in place.
pub fn ifft_1d_complex_inplace(data: &mut Array1<Complex64>) {
    apollo::ifft_1d_complex_inplace(data);
}

/// Forward 1-D complex FFT over a dense slice.
///
/// This preserves the Kwavers `eunomia::Complex64` boundary while Apollo
/// owns Leto/eunomia-native execution internally.
pub fn fft_1d_complex_slice_inplace(data: &mut [Complex64]) {
    <f64 as PlanCacheProvider>::get_1d_plan(Shape1D { n: data.len() })
        .forward_complex_slice_inplace(data);
}

/// Inverse 1-D complex FFT over a dense slice.
///
/// This is the slice counterpart of [`ifft_1d_complex_inplace`].
pub fn ifft_1d_complex_slice_inplace(data: &mut [Complex64]) {
    <f64 as PlanCacheProvider>::get_1d_plan(Shape1D { n: data.len() })
        .inverse_complex_slice_inplace(data);
}

/// Forward 2-D FFT of a real Leto array.
#[must_use]
pub fn fft_2d_array(field: &Array2<f64>) -> Array2<Complex64> {
    apollo::fft_2d_array(field)
}

/// Inverse 2-D FFT of a complex Leto array, returning the real component.
#[must_use]
pub fn ifft_2d_array(field_hat: &Array2<Complex64>) -> Array2<f64> {
    apollo::ifft_2d_array(field_hat)
}

/// Forward 2-D complex FFT, allocating output.
#[must_use]
pub fn fft_2d_complex(field: &Array2<Complex64>) -> Array2<Complex64> {
    let mut out = field.clone();
    fft_2d_complex_inplace(&mut out);
    out
}

/// Inverse 2-D complex FFT, allocating output.
#[must_use]
pub fn ifft_2d_complex(field_hat: &Array2<Complex64>) -> Array2<Complex64> {
    let mut out = field_hat.clone();
    ifft_2d_complex_inplace(&mut out);
    out
}

/// Forward 2-D complex FFT in place.
pub fn fft_2d_complex_inplace(data: &mut Array2<Complex64>) {
    apollo::fft_2d_complex_inplace(data);
}

/// Inverse 2-D complex FFT in place.
pub fn ifft_2d_complex_inplace(data: &mut Array2<Complex64>) {
    apollo::ifft_2d_complex_inplace(data);
}

/// Forward 3-D FFT of a real Leto array.
#[must_use]
pub fn fft_3d_array(field: &Array3<f64>) -> Array3<Complex64> {
    apollo::fft_3d_array(field)
}

/// Forward 3-D FFT of a real Leto array into caller-owned storage.
/// Routes to Apollo's zero-alloc `fft_3d_array_into`, avoiding intermediate
/// allocation and element-wise conversion.
pub fn fft_3d_array_into(field: &Array3<f64>, out: &mut Array3<Complex64>) {
    assert_eq!(
        field.shape(),
        out.shape(),
        "fft_3d_array_into: input and output shapes must match"
    );
    apollo::fft_3d_array_into(field, out);
}

/// Inverse 3-D FFT of a complex Leto array, returning the real component.
#[must_use]
pub fn ifft_3d_array(field_hat: &Array3<Complex64>) -> Array3<f64> {
    apollo::ifft_3d_array(field_hat)
}

/// Inverse 3-D FFT into caller-owned real storage.
/// Routes to Apollo's zero-alloc `ifft_3d_array_into`, avoiding intermediate
/// allocation and element-wise copy.
pub fn ifft_3d_array_into(field_hat: &mut Array3<Complex64>, out: &mut Array3<f64>) {
    assert_eq!(
        field_hat.shape(),
        out.shape(),
        "ifft_3d_array_into: input and output shapes must match"
    );
    apollo::ifft_3d_array_into(field_hat, out);
}

/// Forward 3-D complex FFT, allocating output.
#[must_use]
pub fn fft_3d_complex(field: &Array3<Complex64>) -> Array3<Complex64> {
    let mut out = field.clone();
    fft_3d_complex_inplace(&mut out);
    out
}

/// Forward 3-D complex FFT into caller-owned storage.
pub fn fft_3d_complex_into(field: &Array3<Complex64>, out: &mut Array3<Complex64>) {
    assert_eq!(
        field.shape(),
        out.shape(),
        "fft_3d_complex_into: input and output shapes must match"
    );
    out.assign(field);
    fft_3d_complex_inplace(out);
}

/// Inverse 3-D complex FFT, allocating output.
#[must_use]
pub fn ifft_3d_complex(field_hat: &Array3<Complex64>) -> Array3<Complex64> {
    let mut out = field_hat.clone();
    ifft_3d_complex_inplace(&mut out);
    out
}

/// Forward 3-D complex FFT in place.
pub fn fft_3d_complex_inplace(data: &mut Array3<Complex64>) {
    apollo::fft_3d_complex_inplace(data);
}

/// Inverse 3-D complex FFT in place.
pub fn ifft_3d_complex_inplace(data: &mut Array3<Complex64>) {
    apollo::ifft_3d_complex_inplace(data);
}

/// Forward 3-D complex FFT along one axis in place.
///
/// Kwavers and Apollo share Leto storage and `eunomia::Complex64`, so this
/// delegates directly without allocating or converting the caller's field.
pub fn fft_3d_axis_complex_inplace(plan: &Fft3d, data: &mut Array3<Complex64>, axis: usize) {
    plan.forward_axis_complex_inplace(data, axis);
}

/// Inverse 3-D complex FFT along one axis in place.
///
/// This preserves the direct zero-copy contract of
/// [`fft_3d_axis_complex_inplace`].
pub fn ifft_3d_axis_complex_inplace(plan: &Fft3d, data: &mut Array3<Complex64>, axis: usize) {
    plan.inverse_axis_complex_inplace(data, axis);
}

/// Full-spectrum (nx, ny, nz) complex-to-complex 3-D transforms with caller-owned
/// real and complex storage.
///
/// Local Apollo now accepts Leto arrays and `eunomia::Complex64`. This extension
/// trait preserves Kwavers' Leto/`eunomia` spectral contract at one
/// boundary while Apollo remains the single FFT engine.
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
    /// nz/2+1)` of a real field. The facade computes the full complex
    /// transform through Apollo's Leto path and stores the non-redundant
    /// z-spectrum. `half_out` must have shape `(nx, ny, nz/2+1)`.
    fn forward_r2c_into(&self, real: &Array3<f64>, half_out: &mut Array3<Complex64>);

    /// Inverse complex-to-real 3-D FFT from a **half-spectrum** `(nx, ny,
    /// nz/2+1)` into a real field. The facade reconstructs the full Hermitian
    /// spectrum and calls Apollo's full complex inverse path. The `scratch`
    /// argument is retained for call-site compatibility and is unused.
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
            field.shape(),
            out.shape(),
            "Fft2dInOutExt::forward_into: shape mismatch between real input and complex output"
        );
        assign_real_to_complex_2d(field, out);
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
            field_hat.shape(),
            scratch.shape(),
            "Fft2dInOutExt::inverse_into: shape mismatch between complex input and complex scratch"
        );
        debug_assert_eq!(
            field_hat.shape(),
            out.shape(),
            "Fft2dInOutExt::inverse_into: shape mismatch between complex input and real output"
        );
        scratch.assign(field_hat);
        self.inverse_complex_inplace(scratch);
        assign_complex_real_2d(scratch, out);
    }
}

impl Fft3dInOutExt for Fft3d {
    #[inline]
    fn forward_into(&self, field: &Array3<f64>, out: &mut Array3<Complex64>) {
        debug_assert_eq!(
            field.shape(),
            out.shape(),
            "Fft3dInOutExt::forward_into: shape mismatch between real input and complex output"
        );
        assign_real_to_complex_3d(field, out);
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
            field_hat.shape(),
            scratch.shape(),
            "Fft3dInOutExt::inverse_into: shape mismatch between complex input and complex scratch"
        );
        debug_assert_eq!(
            field_hat.shape(),
            out.shape(),
            "Fft3dInOutExt::inverse_into: shape mismatch between complex input and real output"
        );
        scratch.assign(field_hat);
        self.inverse_complex_inplace(scratch);
        assign_complex_real_3d(scratch, out);
    }

    #[inline]
    fn forward_r2c_into(&self, real: &Array3<f64>, half_out: &mut Array3<Complex64>) {
        let [nx, ny, nz] = real.shape();
        let nz_c = nz / 2 + 1;
        debug_assert_eq!(
            half_out.shape(),
            [nx, ny, nz_c],
            "forward_r2c_into: half_out must be (nx, ny, nz/2+1)"
        );
        R2C_FULL_SCRATCH.with(|cell| {
            let mut borrow = cell.borrow_mut();
            if borrow.shape() != [nx, ny, nz] {
                *borrow = Array3::<Complex64>::from_elem([nx, ny, nz], Complex64::default());
            }
            let full: &mut Array3<Complex64> = &mut borrow;
            assign_real_to_complex_3d(real, full);
            self.forward_complex_inplace(full);
            if let Some(half_values) = half_out.as_slice_mut() {
                let full_values = full
                    .as_slice()
                    .expect("invariant: thread-local FFT scratch is contiguous");
                for (full_row, half_row) in full_values
                    .chunks_exact(nz)
                    .zip(half_values.chunks_exact_mut(nz_c))
                {
                    half_row.copy_from_slice(&full_row[..nz_c]);
                }
            } else {
                half_out.assign(
                    &full
                        .slice(&[(0, nx, 1), (0, ny, 1), (0, nz_c, 1)])
                        .expect("invariant: validated FFT half-spectrum slice"),
                );
            }
        });
    }

    #[inline]
    fn inverse_c2r_into(
        &self,
        half_in: &Array3<Complex64>,
        out: &mut Array3<f64>,
        _scratch: &mut Array3<Complex64>,
    ) {
        let [nx, ny, nz] = out.shape();
        let nz_c = nz / 2 + 1;
        debug_assert_eq!(
            half_in.shape(),
            [nx, ny, nz_c],
            "inverse_c2r_into: half_in must be (nx, ny, nz/2+1)"
        );
        R2C_FULL_SCRATCH.with(|cell| {
            let mut borrow = cell.borrow_mut();
            if borrow.shape() != [nx, ny, nz] {
                *borrow = Array3::<Complex64>::from_elem([nx, ny, nz], Complex64::default());
            }
            let full: &mut Array3<Complex64> = &mut borrow;
            if let Some(half_values) = half_in.as_slice() {
                let full_values = full
                    .as_slice_mut()
                    .expect("invariant: thread-local FFT scratch is contiguous");
                for i in 0..nx {
                    let ii = if i == 0 { 0 } else { nx - i };
                    for j in 0..ny {
                        let jj = if j == 0 { 0 } else { ny - j };
                        let full_row_start = (i * ny + j) * nz;
                        let half_row_start = (i * ny + j) * nz_c;
                        full_values[full_row_start..full_row_start + nz_c]
                            .copy_from_slice(&half_values[half_row_start..half_row_start + nz_c]);

                        let mirror_row_start = (ii * ny + jj) * nz_c;
                        for k in nz_c..nz {
                            full_values[full_row_start + k] =
                                half_values[mirror_row_start + nz - k].conj();
                        }
                    }
                }
            } else {
                full.slice_mut(&[(0, nx, 1), (0, ny, 1), (0, nz_c, 1)])
                    .expect("invariant: validated FFT half-spectrum slice")
                    .assign(half_in);
                for k in nz_c..nz {
                    let kk = nz - k;
                    for i in 0..nx {
                        let ii = if i == 0 { 0 } else { nx - i };
                        for j in 0..ny {
                            let jj = if j == 0 { 0 } else { ny - j };
                            full[[i, j, k]] = half_in[[ii, jj, kk]].conj();
                        }
                    }
                }
            }
            self.inverse_complex_inplace(full);
            assign_complex_real_3d(full, out);
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

fn assign_real_to_complex_2d(real: &Array2<f64>, complex: &mut Array2<Complex64>) {
    assert_eq!(
        real.shape(),
        complex.shape(),
        "real and complex 2-D FFT arrays must have equal shapes"
    );

    if let (Some(real_values), Some(complex_values)) = (real.as_slice(), complex.as_slice_mut()) {
        assign_real_slice_to_complex(real_values, complex_values);
        return;
    }

    for ([i, j], &real_value) in real.indexed_iter() {
        complex[[i, j]] = Complex64::new(real_value, 0.0);
    }
}

fn assign_real_to_complex_3d(real: &Array3<f64>, complex: &mut Array3<Complex64>) {
    assert_eq!(
        real.shape(),
        complex.shape(),
        "real and complex 3-D FFT arrays must have equal shapes"
    );

    if let (Some(real_values), Some(complex_values)) = (real.as_slice(), complex.as_slice_mut()) {
        assign_real_slice_to_complex(real_values, complex_values);
        return;
    }

    for ([i, j, k], &real_value) in real.indexed_iter() {
        complex[[i, j, k]] = Complex64::new(real_value, 0.0);
    }
}

fn assign_complex_real_2d(complex: &Array2<Complex64>, real: &mut Array2<f64>) {
    assert_eq!(
        complex.shape(),
        real.shape(),
        "complex and real 2-D FFT arrays must have equal shapes"
    );

    if let (Some(complex_values), Some(real_values)) = (complex.as_slice(), real.as_slice_mut()) {
        assign_complex_slice_real(complex_values, real_values);
        return;
    }

    for ([i, j], complex_value) in complex.indexed_iter() {
        real[[i, j]] = complex_value.re;
    }
}

fn assign_complex_real_3d(complex: &Array3<Complex64>, real: &mut Array3<f64>) {
    assert_eq!(
        complex.shape(),
        real.shape(),
        "complex and real 3-D FFT arrays must have equal shapes"
    );

    if let (Some(complex_values), Some(real_values)) = (complex.as_slice(), real.as_slice_mut()) {
        assign_complex_slice_real(complex_values, real_values);
        return;
    }

    for ([i, j, k], complex_value) in complex.indexed_iter() {
        real[[i, j, k]] = complex_value.re;
    }
}

fn assign_real_slice_to_complex(real_values: &[f64], complex_values: &mut [Complex64]) {
    for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
        complex_values,
        FFT_ASSIGN_CHUNK_LEN,
        |chunk_index, chunk| {
            let base = chunk_index * FFT_ASSIGN_CHUNK_LEN;
            for (offset, complex_value) in chunk.iter_mut().enumerate() {
                *complex_value = Complex64::new(real_values[base + offset], 0.0);
            }
        },
    );
}

fn assign_complex_slice_real(complex_values: &[Complex64], real_values: &mut [f64]) {
    for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
        real_values,
        FFT_ASSIGN_CHUNK_LEN,
        |chunk_index, chunk| {
            let base = chunk_index * FFT_ASSIGN_CHUNK_LEN;
            for (offset, real_value) in chunk.iter_mut().enumerate() {
                *real_value = complex_values[base + offset].re;
            }
        },
    );
}

#[cfg(test)]
mod r2c_optimized_tests {
    use super::{fft_3d_complex_inplace, get_fft_for_grid, Fft3dInOutExt};
    use eunomia::Complex64;
    use leto::Array3;

    fn check_shape(nx: usize, ny: usize, nz: usize) {
        let nz_c = nz / 2 + 1;
        let fft = get_fft_for_grid(nx, ny, nz);
        let real = Array3::from_shape_fn([nx, ny, nz], |[i, j, k]| {
            let x = ((i * 131 + j * 17 + k * 7) % 101) as f64 / 101.0 - 0.5;
            (x * std::f64::consts::TAU).sin() + 0.3 * x + 0.1
        });

        // (1) forward_r2c is bit-identical to a full c2c + truncation.
        let mut half_new = Array3::zeros([nx, ny, nz_c]);
        fft.forward_r2c_into(&real, &mut half_new);
        let mut full = real.mapv(|v| Complex64::new(v, 0.0));
        fft_3d_complex_inplace(&mut full);
        let ref_half = full.slice(&[(0, nx, 1), (0, ny, 1), (0, nz_c, 1)]).unwrap();
        let fwd_err = half_new
            .iter()
            .zip(ref_half.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(
            fwd_err < 1e-9,
            "forward_r2c({nx},{ny},{nz}) vs full-c2c reference: {fwd_err:.2e}"
        );

        // (2) inverse_c2r recovers the real field (round-trip).
        let mut out = Array3::zeros([nx, ny, nz]);
        let mut scratch = Array3::zeros([nx, ny, nz_c]);
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
