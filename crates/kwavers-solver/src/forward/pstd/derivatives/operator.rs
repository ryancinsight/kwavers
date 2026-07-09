//! Spectral Derivative Operators for Pseudospectral Methods
//!
//! ## Theorem (Spectral Derivative via DFT)
//!
//! **Statement** (Trefethen 2000, Thm. 3.1): Let `u[n]` be a real N-periodic
//! grid function sampled at `xₙ = n·Δx`, `n = 0, …, N−1`. Define the DFT and
//! IDFT pair in the usual convention. The **spectral derivative** is:
//!
//! ```text
//! ∂u/∂x ≈ F⁻¹[ i·ω[k] · F[u] ]
//! ```
//!
//! where the wavenumber array is (for N even):
//! ```text
//! ω[k] = 2πk/(N·Δx)   for k = 0, 1, …, N/2−1
//! ω[k] = 2π(k−N)/(N·Δx) for k = N/2, …, N−1   (negative frequencies)
//! ω[N/2] = 0           (Nyquist mode zeroed — no alias-free derivative exists)
//! ```
//!
//! **Exactness:** For a DFT-representable mode `u[n] = A·sin(2πm·n/N)`,
//! `m ∈ {1, …, N/2−1}`, the spectral derivative recovers `A·ω[m]·cos(2πm·n/N)`
//! to within floating-point rounding (~O(N·log₂(N)·ε_mach) ≈ 10⁻¹³ for N=32).
//! No aliasing occurs because m < N/2.
//!
//! **Spectral accuracy:** For analytic periodic functions the truncation error
//! decays faster than any polynomial in Δx (exponential convergence).
//! For functions with `p` continuous derivatives, the error is O(Δx^p).
//!
//! ## Theorem (2/3-Rule Dealiasing)
//!
//! **Statement** (Orszag 1971): When nonlinear products `p(x)·q(x)` generate
//! modes up to `2kₘₐₓ` (where `kₘₐₓ = π/Δx` is the Nyquist limit), aliases
//! fold back at `k = 2kₘₐₓ − k'`. To prevent aliasing from corrupting the
//! resolved modes `k ≤ kₘₐₓ/2`, zero all modes with:
//! ```text
//! |ω[k]| > 2π/(3Δx)   (2/3 of the Nyquist wavenumber)
//! ```
//! This ensures that when two dealiased fields of bandwidth `2π/(3Δx)` are
//! multiplied, their product has bandwidth `4π/(3Δx) < 2π/Δx = kₙᵧq`, so
//! no aliases enter the retained band.
//!
//! For N=32 and mode m=1: ω[1] = 2π/(32·Δx) ≪ 2π/(3Δx), so the fundamental
//! mode passes trivially.
//!
//! ## References
//!
//! - Trefethen LN (2000). Spectral Methods in MATLAB. SIAM. Thm. 3.1.
//! - Orszag SA (1971). "On the elimination of aliasing in FD schemes by
//!   filtering high-wavenumber components." J. Atmos. Sci. 28(6), 1074.
//! - Boyd JP (2001). Chebyshev and Fourier Spectral Methods. Dover. §3.
//! - Canuto C et al. (2006). Spectral Methods: Fundamentals in Single Domains.
//!   Springer. §2.3.

use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::{fft_1d_complex_inplace, ifft_1d_complex_inplace, Complex64};
use leto::Array1 as LetoArray1;
use moirai_parallel::{for_each_chunk_mut_enumerated_with, map_collect_index_with, Adaptive};
use leto::{
    Array1,
    Array3,
    ArrayView3,
};

/// Spectral derivative operator for 3D fields.
///
/// The i*k*dealiasing multipliers are pre-computed at construction; FFT plan
/// caching is owned by the `kwavers_math::fft` facade.
pub struct SpectralDerivativeOperator {
    pub(super) nx: usize,
    pub(super) ny: usize,
    pub(super) nz: usize,

    ikd_x: Vec<Complex64>,
    ikd_y: Vec<Complex64>,
    ikd_z: Vec<Complex64>,
}

impl std::fmt::Debug for SpectralDerivativeOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpectralDerivativeOperator")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("nz", &self.nz)
            .finish()
    }
}

impl Clone for SpectralDerivativeOperator {
    fn clone(&self) -> Self {
        Self {
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
            ikd_x: self.ikd_x.clone(),
            ikd_y: self.ikd_y.clone(),
            ikd_z: self.ikd_z.clone(),
        }
    }
}

impl SpectralDerivativeOperator {
    /// Create new spectral derivative operator.
    ///
    /// # Panics
    ///
    /// If any grid dimension is 0.
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> Self {
        assert!(
            nx > 0 && ny > 0 && nz > 0,
            "Grid dimensions must be positive"
        );

        let kx = Self::compute_wavenumbers(nx, dx);
        let ky = Self::compute_wavenumbers(ny, dy);
        let kz = Self::compute_wavenumbers(nz, dz);

        let dealiasing_filter_x = Self::compute_dealiasing_filter(nx, dx);
        let dealiasing_filter_y = Self::compute_dealiasing_filter(ny, dy);
        let dealiasing_filter_z = Self::compute_dealiasing_filter(nz, dz);

        let ikd_x: Vec<Complex64> = (0..nx)
            .map(|i| Complex64::new(0.0, kx[i] * dealiasing_filter_x[i]))
            .collect();
        let ikd_y: Vec<Complex64> = (0..ny)
            .map(|i| Complex64::new(0.0, ky[i] * dealiasing_filter_y[i]))
            .collect();
        let ikd_z: Vec<Complex64> = (0..nz)
            .map(|i| Complex64::new(0.0, kz[i] * dealiasing_filter_z[i]))
            .collect();

        Self {
            nx,
            ny,
            nz,
            ikd_x,
            ikd_y,
            ikd_z,
        }
    }

    /// Compute wavenumber array: k[n] = 2π·n/(N·Δx) for n < N/2, 2π·(n-N)/(N·Δx) otherwise.
    fn compute_wavenumbers(n: usize, dx: f64) -> Array1<f64> {
        let mut k = Array1::zeros(n);
        let norm = TWO_PI / (n as f64 * dx);
        for i in 0..n / 2 {
            k[i] = i as f64 * norm;
        }
        for i in n / 2..n {
            k[i] = (i as f64 - n as f64) * norm;
        }
        k
    }

    /// Compute 2/3-rule dealiasing filter (sets |k| > 2π/(3Δx) to zero).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_dealiasing_filter(n: usize, dx: f64) -> Array1<f64> {
        let mut filter = Array1::ones(n);
        let cutoff = TWO_PI / (3.0 * dx);
        let k = Self::compute_wavenumbers(n, dx);
        for i in 0..n {
            if k[i].abs() > cutoff {
                filter[i] = 0.0;
            }
        }
        filter
    }

    /// Compute x-derivative via spectral method (FFT along x pencils).
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn derivative_x(&self, field: &ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        self.validate_field(field)?;
        let mut derivative = Array3::zeros([self.nx, self.ny, self.nz]);
        self.derivative_along_x_impl(field, &mut derivative)?;
        self.validate_output(&derivative)?;
        Ok(derivative)
    }

    /// Compute y-derivative via spectral method (FFT along y pencils).
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn derivative_y(&self, field: &ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        self.validate_field(field)?;
        let mut derivative = Array3::zeros([self.nx, self.ny, self.nz]);
        self.derivative_along_y_impl(field, &mut derivative)?;
        self.validate_output(&derivative)?;
        Ok(derivative)
    }

    /// Compute z-derivative via spectral method (FFT along z pencils).
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn derivative_z(&self, field: &ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        self.validate_field(field)?;
        let mut derivative = Array3::zeros([self.nx, self.ny, self.nz]);
        self.derivative_along_z_impl(field, &mut derivative)?;
        self.validate_output(&derivative)?;
        Ok(derivative)
    }

    /// In-place x-derivative into a preallocated `out` (no allocation). The
    /// finiteness scan of [`Self::derivative_x`] is skipped for hot-path reuse;
    /// `out` must already have the grid shape.
    /// # Errors
    /// - Shape mismatch of `field` or `out`.
    pub fn derivative_x_into(
        &self,
        field: &ArrayView3<f64>,
        out: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        self.check_shapes(field, out)?;
        self.derivative_along_x_impl(field, out)
    }

    /// In-place y-derivative into a preallocated `out`. See [`Self::derivative_x_into`].
    /// # Errors
    /// - Shape mismatch of `field` or `out`.
    pub fn derivative_y_into(
        &self,
        field: &ArrayView3<f64>,
        out: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        self.check_shapes(field, out)?;
        self.derivative_along_y_impl(field, out)
    }

    /// In-place z-derivative into a preallocated `out`. See [`Self::derivative_x_into`].
    /// # Errors
    /// - Shape mismatch of `field` or `out`.
    pub fn derivative_z_into(
        &self,
        field: &ArrayView3<f64>,
        out: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        self.check_shapes(field, out)?;
        self.derivative_along_z_impl(field, out)
    }

    #[inline]
    fn check_shapes(&self, field: &ArrayView3<f64>, out: &Array3<f64>) -> KwaversResult<()> {
        let expected = [self.nx, self.ny, self.nz];
        if field.shape() != expected || out.shape() != expected {
            return Err(KwaversError::InvalidInput(format!(
                "derivative shape mismatch: field {:?}, out {:?}, grid {:?}",
                field.shape(),
                out.shape(),
                expected
            )));
        }
        Ok(())
    }

    #[inline]
    fn validate_field(&self, field: &ArrayView3<f64>) -> KwaversResult<()> {
        if field.shape() != [self.nx, self.ny, self.nz] {
            return Err(KwaversError::InvalidInput(format!(
                "Field shape {:?} mismatch grid {:?}",
                field.shape(),
                &[self.nx, self.ny, self.nz]
            )));
        }
        if !field.iter().all(|&x| x.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "Input field contains NaN or Inf values".into(),
            ));
        }
        Ok(())
    }

    #[inline]
    fn validate_output(&self, derivative: &Array3<f64>) -> KwaversResult<()> {
        if !derivative.iter().all(|&x| x.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "Output field contains NaN or Inf values (numerical instability)".into(),
            ));
        }
        Ok(())
    }

    /// Schedules one `(j, k)` pencil per Moirai task and scatters the x-line
    /// results back into the strided output field.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn derivative_along_x_impl(
        &self,
        field: &ArrayView3<f64>,
        derivative: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let nx = self.nx;
        let nz = self.nz;
        let ikd = &self.ikd_x;

        let pencil_count = self.ny * nz;
        let pencils = map_collect_index_with::<Adaptive, _, _>(pencil_count, |pencil_index| {
            let j = pencil_index / nz;
            let l = pencil_index % nz;
            let mut line = LetoArray1::<Complex64>::from_elem([nx], Complex64::default());
            for i in 0..nx {
                line[i] = Complex64::new(field[[i, j, l]], 0.0);
            }
            fft_1d_complex_inplace(&mut line);
            for (i, &ikd_val) in ikd.iter().enumerate() {
                line[i] *= ikd_val;
            }
            ifft_1d_complex_inplace(&mut line);
            let values: Vec<f64> = line.iter().map(|value| value.re).collect();
            (j, l, values)
        });

        for (j, l, values) in pencils {
            for (i, value) in values.into_iter().enumerate() {
                derivative[[i, j, l]] = value;
            }
        }

        Ok(())
    }

    /// Schedules one contiguous i-slab per Moirai task.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn derivative_along_y_impl(
        &self,
        field: &ArrayView3<f64>,
        derivative: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let ny = self.ny;
        let nz = self.nz;
        let ikd = &self.ikd_y;

        if let Some(derivative_values) = derivative.as_slice_memory_order_mut() {
            let slab_len = ny * nz;
            for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                derivative_values,
                slab_len,
                |i, slab| {
                    let mut line = LetoArray1::<Complex64>::from_elem([ny], Complex64::default());
                    for l in 0..nz {
                        for j in 0..ny {
                            line[j] = Complex64::new(field[[i, j, l]], 0.0);
                        }
                        fft_1d_complex_inplace(&mut line);
                        for (j, &ikd_val) in ikd.iter().enumerate() {
                            line[j] *= ikd_val;
                        }
                        ifft_1d_complex_inplace(&mut line);
                        for j in 0..ny {
                            slab[j * nz + l] = line[j].re;
                        }
                    }
                },
            );
        } else {
            for i in 0..self.nx {
                let mut line = LetoArray1::<Complex64>::from_elem([ny], Complex64::default());
                for l in 0..nz {
                    for j in 0..ny {
                        line[j] = Complex64::new(field[[i, j, l]], 0.0);
                    }
                    fft_1d_complex_inplace(&mut line);
                    for (j, &ikd_val) in ikd.iter().enumerate() {
                        line[j] *= ikd_val;
                    }
                    ifft_1d_complex_inplace(&mut line);
                    for j in 0..ny {
                        derivative[[i, j, l]] = line[j].re;
                    }
                }
            }
        }

        Ok(())
    }

    /// Schedules one contiguous i-slab per Moirai task.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn derivative_along_z_impl(
        &self,
        field: &ArrayView3<f64>,
        derivative: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let ny = self.ny;
        let nz = self.nz;
        let ikd = &self.ikd_z;

        if let Some(derivative_values) = derivative.as_slice_memory_order_mut() {
            let slab_len = ny * nz;
            for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                derivative_values,
                slab_len,
                |i, slab| {
                    let mut line = LetoArray1::<Complex64>::from_elem([nz], Complex64::default());
                    for j in 0..ny {
                        for l in 0..nz {
                            line[l] = Complex64::new(field[[i, j, l]], 0.0);
                        }
                        fft_1d_complex_inplace(&mut line);
                        for (l, &ikd_val) in ikd.iter().enumerate() {
                            line[l] *= ikd_val;
                        }
                        ifft_1d_complex_inplace(&mut line);
                        let row_start = j * nz;
                        for l in 0..nz {
                            slab[row_start + l] = line[l].re;
                        }
                    }
                },
            );
        } else {
            for i in 0..self.nx {
                let mut line = LetoArray1::<Complex64>::from_elem([nz], Complex64::default());
                for j in 0..ny {
                    for l in 0..nz {
                        line[l] = Complex64::new(field[[i, j, l]], 0.0);
                    }
                    fft_1d_complex_inplace(&mut line);
                    for (l, &ikd_val) in ikd.iter().enumerate() {
                        line[l] *= ikd_val;
                    }
                    ifft_1d_complex_inplace(&mut line);
                    for l in 0..nz {
                        derivative[[i, j, l]] = line[l].re;
                    }
                }
            }
        }

        Ok(())
    }
}
