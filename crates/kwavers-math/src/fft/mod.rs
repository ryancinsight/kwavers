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

use apollo::Complex64 as ApolloComplex64;
pub use apollo::{
    fftfreq, fftshift, ifftshift, rfftfreq, FftPlan1D, FftPlan2D, FftPlan3D, Normalization,
    PlanCacheProvider, Shape1D, Shape2D, Shape3D,
};
pub use num_complex::{Complex32, Complex64};
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

use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};
use ndarray::{s, Array1, Array2, Array3, ArrayBase, Data, DataMut, Dimension, Zip};
use std::cell::RefCell;

const FFT_ASSIGN_CHUNK_LEN: usize = 4096;

thread_local! {
    /// Per-thread full-spectrum `(nx, ny, nz)` complex scratch used by the
    /// half-spectrum r2c/c2r emulation in [`Fft3dInOutExt`]. Apollo dropped its
    /// public half-spectrum transforms; this scratch lets the ACL run apollo's
    /// full-spectrum complex plan and truncate/expand to the `nz_c = nz/2 + 1`
    /// layout the PSTD core still uses. Resized on grid-shape change, then reused
    /// across timesteps (zero steady-state allocation for a fixed grid).
    static R2C_FULL_SCRATCH: RefCell<Array3<Complex64>> = RefCell::new(Array3::zeros((0, 0, 0)));
}

/// Forward 1-D FFT of a real ndarray through Apollo's Leto-owned engine.
#[must_use]
pub fn fft_1d_array(field: &Array1<f64>) -> Array1<Complex64> {
    from_leto_complex_1d(&apollo::fft_1d_array(&to_leto_real_1d(field)))
}

/// Inverse 1-D FFT of a complex ndarray, returning the real component.
#[must_use]
pub fn ifft_1d_array(field_hat: &Array1<Complex64>) -> Array1<f64> {
    from_leto_real_1d(&apollo::ifft_1d_array(&to_leto_complex_1d(field_hat)))
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
    let mut leto_data = to_leto_complex_1d(data);
    apollo::fft_1d_complex_inplace(&mut leto_data);
    copy_leto_complex_1d(&leto_data, data);
}

/// Inverse 1-D complex FFT in place.
pub fn ifft_1d_complex_inplace(data: &mut Array1<Complex64>) {
    let mut leto_data = to_leto_complex_1d(data);
    apollo::ifft_1d_complex_inplace(&mut leto_data);
    copy_leto_complex_1d(&leto_data, data);
}

/// Forward 1-D complex FFT over a dense slice.
///
/// This preserves the Kwavers `num_complex::Complex64` boundary while Apollo
/// owns Leto/eunomia-native execution internally.
pub fn fft_1d_complex_slice_inplace(data: &mut [Complex64]) {
    let mut leto_data = to_leto_complex_slice(data);
    apollo::fft_1d_complex_inplace(&mut leto_data);
    copy_leto_complex_slice(&leto_data, data);
}

/// Inverse 1-D complex FFT over a dense slice.
///
/// This is the slice counterpart of [`ifft_1d_complex_inplace`].
pub fn ifft_1d_complex_slice_inplace(data: &mut [Complex64]) {
    let mut leto_data = to_leto_complex_slice(data);
    apollo::ifft_1d_complex_inplace(&mut leto_data);
    copy_leto_complex_slice(&leto_data, data);
}

/// Forward 2-D FFT of a real ndarray.
#[must_use]
pub fn fft_2d_array(field: &Array2<f64>) -> Array2<Complex64> {
    let mut out = field.mapv(|value| Complex64::new(value, 0.0));
    fft_2d_complex_inplace(&mut out);
    out
}

/// Inverse 2-D FFT of a complex ndarray, returning the real component.
#[must_use]
pub fn ifft_2d_array(field_hat: &Array2<Complex64>) -> Array2<f64> {
    let mut tmp = field_hat.clone();
    ifft_2d_complex_inplace(&mut tmp);
    tmp.mapv(|value| value.re)
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
    let mut leto_data = to_leto_complex_2d(data);
    apollo::fft_2d_complex_inplace(&mut leto_data);
    copy_leto_complex_2d(&leto_data, data);
}

/// Inverse 2-D complex FFT in place.
pub fn ifft_2d_complex_inplace(data: &mut Array2<Complex64>) {
    let mut leto_data = to_leto_complex_2d(data);
    apollo::ifft_2d_complex_inplace(&mut leto_data);
    copy_leto_complex_2d(&leto_data, data);
}

/// Forward 3-D FFT of a real ndarray.
#[must_use]
pub fn fft_3d_array(field: &Array3<f64>) -> Array3<Complex64> {
    from_leto_complex_3d(&apollo::fft_3d_array(&to_leto_real_3d(field)))
}

/// Forward 3-D FFT of a real ndarray into caller-owned storage.
pub fn fft_3d_array_into(field: &Array3<f64>, out: &mut Array3<Complex64>) {
    assert_eq!(
        field.dim(),
        out.dim(),
        "fft_3d_array_into: input and output shapes must match"
    );
    out.assign(&fft_3d_array(field));
}

/// Inverse 3-D FFT of a complex ndarray, returning the real component.
#[must_use]
pub fn ifft_3d_array(field_hat: &Array3<Complex64>) -> Array3<f64> {
    from_leto_real_3d(&apollo::ifft_3d_array(&to_leto_complex_3d(field_hat)))
}

/// Inverse 3-D FFT into caller-owned real storage.
pub fn ifft_3d_array_into(field_hat: &mut Array3<Complex64>, out: &mut Array3<f64>) {
    assert_eq!(
        field_hat.dim(),
        out.dim(),
        "ifft_3d_array_into: input and output shapes must match"
    );
    out.assign(&ifft_3d_array(field_hat));
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
        field.dim(),
        out.dim(),
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
    let mut leto_data = to_leto_complex_3d(data);
    apollo::fft_3d_complex_inplace(&mut leto_data);
    copy_leto_complex_3d(&leto_data, data);
}

/// Inverse 3-D complex FFT in place.
pub fn ifft_3d_complex_inplace(data: &mut Array3<Complex64>) {
    let mut leto_data = to_leto_complex_3d(data);
    apollo::ifft_3d_complex_inplace(&mut leto_data);
    copy_leto_complex_3d(&leto_data, data);
}

/// Forward 3-D complex FFT along one axis in place.
///
/// This preserves the ndarray-facing Kwavers contract while Apollo owns the
/// Leto-backed axis transform implementation.
pub fn fft_3d_axis_complex_inplace(plan: &Fft3d, data: &mut Array3<Complex64>, axis: usize) {
    let mut leto_data = to_leto_complex_3d(data);
    plan.forward_axis_complex_inplace(&mut leto_data, axis);
    copy_leto_complex_3d(&leto_data, data);
}

/// Inverse 3-D complex FFT along one axis in place.
///
/// This is the axis-transform counterpart to [`ifft_3d_complex_inplace`].
pub fn ifft_3d_axis_complex_inplace(plan: &Fft3d, data: &mut Array3<Complex64>, axis: usize) {
    let mut leto_data = to_leto_complex_3d(data);
    plan.inverse_axis_complex_inplace(&mut leto_data, axis);
    copy_leto_complex_3d(&leto_data, data);
}

/// Full-spectrum (nx, ny, nz) complex-to-complex 3-D transforms with caller-owned
/// real and complex storage.
///
/// Local Apollo now accepts Leto arrays and `eunomia::Complex64`. This extension
/// trait preserves Kwavers' ndarray/`num_complex` spectral contract at one
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
            field.dim(),
            out.dim(),
            "Fft2dInOutExt::forward_into: shape mismatch between real input and complex output"
        );
        assign_real_to_complex(field, out);
        fft_2d_complex_inplace(out);
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
        ifft_2d_complex_inplace(scratch);
        assign_complex_real(scratch, out);
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
        assign_real_to_complex(field, out);
        fft_3d_complex_inplace(out);
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
        ifft_3d_complex_inplace(scratch);
        assign_complex_real(scratch, out);
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
        R2C_FULL_SCRATCH.with(|cell| {
            let mut borrow = cell.borrow_mut();
            if borrow.dim() != (nx, ny, nz) {
                *borrow = Array3::<Complex64>::zeros((nx, ny, nz));
            }
            let full: &mut Array3<Complex64> = &mut borrow;
            assign_real_to_complex(real, full);
            fft_3d_complex_inplace(full);
            half_out.assign(&full.slice(s![.., .., 0..nz_c]));
        });
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
        R2C_FULL_SCRATCH.with(|cell| {
            let mut borrow = cell.borrow_mut();
            if borrow.dim() != (nx, ny, nz) {
                *borrow = Array3::<Complex64>::zeros((nx, ny, nz));
            }
            let full: &mut Array3<Complex64> = &mut borrow;
            full.slice_mut(s![.., .., 0..nz_c]).assign(half_in);
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
            ifft_3d_complex_inplace(full);
            assign_complex_real(full, out);
        });
    }

    #[inline]
    fn forward(&self, real: &Array3<f64>) -> Array3<Complex64> {
        let mut out = real.mapv(|v| Complex64::new(v, 0.0));
        fft_3d_complex_inplace(&mut out);
        out
    }

    #[inline]
    fn inverse(&self, spectrum: &Array3<Complex64>) -> Array3<f64> {
        let mut tmp = spectrum.clone();
        ifft_3d_complex_inplace(&mut tmp);
        tmp.mapv(|c| c.re)
    }
}

fn to_apollo_complex(value: Complex64) -> ApolloComplex64 {
    ApolloComplex64::new(value.re, value.im)
}

fn from_apollo_complex(value: ApolloComplex64) -> Complex64 {
    Complex64::new(value.re, value.im)
}

fn to_leto_real_1d(field: &Array1<f64>) -> leto::Array1<f64> {
    leto::Array1::from_shape_vec([field.len()], field.iter().copied().collect())
        .expect("ndarray 1-D real data must map to a Leto array of equal length")
}

fn to_leto_real_3d(field: &Array3<f64>) -> leto::Array3<f64> {
    let (nx, ny, nz) = field.dim();
    leto::Array3::from_shape_vec([nx, ny, nz], field.iter().copied().collect())
        .expect("ndarray 3-D real data must map to a Leto array of equal shape")
}

fn to_leto_complex_1d(field: &Array1<Complex64>) -> leto::Array1<ApolloComplex64> {
    leto::Array1::from_shape_vec(
        [field.len()],
        field.iter().copied().map(to_apollo_complex).collect(),
    )
    .expect("ndarray 1-D complex data must map to a Leto array of equal length")
}

fn to_leto_complex_slice(field: &[Complex64]) -> leto::Array1<ApolloComplex64> {
    leto::Array1::from_shape_vec(
        [field.len()],
        field.iter().copied().map(to_apollo_complex).collect(),
    )
    .expect("complex slice data must map to a Leto array of equal length")
}

fn to_leto_complex_2d(field: &Array2<Complex64>) -> leto::Array2<ApolloComplex64> {
    let (nx, ny) = field.dim();
    leto::Array2::from_shape_vec(
        [nx, ny],
        field.iter().copied().map(to_apollo_complex).collect(),
    )
    .expect("ndarray 2-D complex data must map to a Leto array of equal shape")
}

fn to_leto_complex_3d(field: &Array3<Complex64>) -> leto::Array3<ApolloComplex64> {
    let (nx, ny, nz) = field.dim();
    leto::Array3::from_shape_vec(
        [nx, ny, nz],
        field.iter().copied().map(to_apollo_complex).collect(),
    )
    .expect("ndarray 3-D complex data must map to a Leto array of equal shape")
}

fn from_leto_real_1d(field: &leto::Array1<f64>) -> Array1<f64> {
    Array1::from_vec(
        field
            .as_slice_memory_order()
            .expect("Leto Array1 from VecStorage must be contiguous")
            .to_vec(),
    )
}

fn from_leto_real_3d(field: &leto::Array3<f64>) -> Array3<f64> {
    let [nx, ny, nz] = field.shape();
    Array3::from_shape_vec(
        (nx, ny, nz),
        field
            .as_slice_memory_order()
            .expect("Leto Array3 from VecStorage must be contiguous")
            .to_vec(),
    )
    .expect("Leto 3-D real data must convert to ndarray with the same shape")
}

fn from_leto_complex_1d(field: &leto::Array1<ApolloComplex64>) -> Array1<Complex64> {
    Array1::from_vec(
        field
            .as_slice_memory_order()
            .expect("Leto Array1 from VecStorage must be contiguous")
            .iter()
            .copied()
            .map(from_apollo_complex)
            .collect(),
    )
}

fn from_leto_complex_3d(field: &leto::Array3<ApolloComplex64>) -> Array3<Complex64> {
    let [nx, ny, nz] = field.shape();
    Array3::from_shape_vec(
        (nx, ny, nz),
        field
            .as_slice_memory_order()
            .expect("Leto Array3 from VecStorage must be contiguous")
            .iter()
            .copied()
            .map(from_apollo_complex)
            .collect(),
    )
    .expect("Leto 3-D complex data must convert to ndarray with the same shape")
}

fn copy_leto_complex_1d(field: &leto::Array1<ApolloComplex64>, out: &mut Array1<Complex64>) {
    assert_eq!(field.shape(), [out.len()]);
    for (dst, src) in out.iter_mut().zip(
        field
            .as_slice_memory_order()
            .expect("Leto Array1 from VecStorage must be contiguous"),
    ) {
        *dst = from_apollo_complex(*src);
    }
}

fn copy_leto_complex_slice(field: &leto::Array1<ApolloComplex64>, out: &mut [Complex64]) {
    assert_eq!(field.shape(), [out.len()]);
    for (dst, src) in out.iter_mut().zip(
        field
            .as_slice_memory_order()
            .expect("Leto Array1 from VecStorage must be contiguous"),
    ) {
        *dst = from_apollo_complex(*src);
    }
}

fn copy_leto_complex_2d(field: &leto::Array2<ApolloComplex64>, out: &mut Array2<Complex64>) {
    let [nx, ny] = field.shape();
    assert_eq!((nx, ny), out.dim());
    for (dst, src) in out.iter_mut().zip(
        field
            .as_slice_memory_order()
            .expect("Leto Array2 from VecStorage must be contiguous"),
    ) {
        *dst = from_apollo_complex(*src);
    }
}

fn copy_leto_complex_3d(field: &leto::Array3<ApolloComplex64>, out: &mut Array3<Complex64>) {
    let [nx, ny, nz] = field.shape();
    assert_eq!((nx, ny, nz), out.dim());
    for (dst, src) in out.iter_mut().zip(
        field
            .as_slice_memory_order()
            .expect("Leto Array3 from VecStorage must be contiguous"),
    ) {
        *dst = from_apollo_complex(*src);
    }
}

fn assign_real_to_complex<Sr, Sc, D>(real: &ArrayBase<Sr, D>, complex: &mut ArrayBase<Sc, D>)
where
    Sr: Data<Elem = f64>,
    Sc: DataMut<Elem = Complex64>,
    D: Dimension,
{
    assert_eq!(
        real.dim(),
        complex.dim(),
        "real and complex FFT arrays must have equal shapes"
    );

    if let (Some(real_values), Some(complex_values)) = (
        real.as_slice_memory_order(),
        complex.as_slice_memory_order_mut(),
    ) {
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
        return;
    }

    Zip::from(complex)
        .and(real)
        .for_each(|complex_value, &real_value| *complex_value = Complex64::new(real_value, 0.0));
}

fn assign_complex_real<Sc, Sr, D>(complex: &ArrayBase<Sc, D>, real: &mut ArrayBase<Sr, D>)
where
    Sc: Data<Elem = Complex64>,
    Sr: DataMut<Elem = f64>,
    D: Dimension,
{
    assert_eq!(
        complex.dim(),
        real.dim(),
        "complex and real FFT arrays must have equal shapes"
    );

    if let (Some(complex_values), Some(real_values)) = (
        complex.as_slice_memory_order(),
        real.as_slice_memory_order_mut(),
    ) {
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
        return;
    }

    Zip::from(real)
        .and(complex)
        .for_each(|real_value, complex_value| *real_value = complex_value.re);
}

#[cfg(test)]
mod r2c_optimized_tests {
    use super::{fft_3d_complex_inplace, get_fft_for_grid, Fft3dInOutExt};
    use ndarray::{s, Array3};
    use num_complex::Complex64;

    fn check_shape(nx: usize, ny: usize, nz: usize) {
        let nz_c = nz / 2 + 1;
        let fft = get_fft_for_grid(nx, ny, nz);
        let real = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            let x = ((i * 131 + j * 17 + k * 7) % 101) as f64 / 101.0 - 0.5;
            (x * std::f64::consts::TAU).sin() + 0.3 * x + 0.1
        });

        // (1) forward_r2c is bit-identical to a full c2c + truncation.
        let mut half_new = Array3::zeros((nx, ny, nz_c));
        fft.forward_r2c_into(&real, &mut half_new);
        let mut full = real.mapv(|v| Complex64::new(v, 0.0));
        fft_3d_complex_inplace(&mut full);
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

        // (2) inverse_c2r recovers the real field (round-trip).
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
