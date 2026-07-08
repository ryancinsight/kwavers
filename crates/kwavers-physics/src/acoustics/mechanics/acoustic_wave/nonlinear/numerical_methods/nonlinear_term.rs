use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_math::fft::Complex64 as Complex;
use kwavers_math::fft::{fft_3d_array, ifft_3d_array};
use kwavers_medium::Medium;
use leto::Array3 as LetoArray3;
use ndarray::Array3;

use super::super::wave_model::NonlinearWave;
use super::array_boundary::{leto_real_field, ndarray_real_field};

impl NonlinearWave {
    /// Computes the nonlinear source term for the Westervelt acoustic wave equation.
    ///
    /// Evaluates Δp_nl = 2β·dt²/ρ₀·[p·∇²p + |∇p|²] per voxel, where β = 1 + B/(2A)
    /// is the coefficient of nonlinearity and ρ₀ is the local medium density.
    ///
    /// ## Anti-aliasing
    ///
    /// A 2/3-rule dealiasing filter is applied in spectral space before any
    /// physical-space product is formed. This prevents the quadratic products
    /// (p·∇²p and |∇p|²) from folding high-wavenumber energy back into the
    /// resolved band. Reference: Canuto et al. (2006) *Spectral Methods*, §3.2.5.
    ///
    /// ## Efficiency
    ///
    /// A single FFT of the pressure field is shared across gradient (×3) and
    /// Laplacian (×1) computations, reducing the forward transform count from
    /// 4 (previous implementation) to 1.
    ///
    /// ## Westervelt approximation
    ///
    /// ∂²(p²)/∂t² ≈ c₀²∇²(p²) = 2c₀²[p∇²p + (∇p)²] (plane-wave, broadband).
    /// Source = (β/ρ₀c₀²)·2c₀²·[p∇²p+(∇p)²] = 2β/ρ₀·[p∇²p+(∇p)²].
    /// Reference: Hamilton & Blackstock (1998) *Nonlinear Acoustics*, Ch. 3.
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    pub(crate) fn compute_nonlinear_term(
        &self,
        pressure: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = pressure.dim();

        // Single FFT of pressure — shared by gradient and Laplacian computations.
        let mut pressure_k = fft_3d_array(&leto_real_field(pressure));

        // 2/3-rule dealiasing: zero bins with absolute frequency index > n/3 along
        // each axis. Physical-space products of the resulting bandlimited fields
        // produce no aliases within the retained [0, 2n/3] wavenumber range.
        Self::apply_dealiasing_filter(&mut pressure_k, nx, ny, nz);

        let kx = grid.compute_kx();
        let ky = grid.compute_ky();
        let kz = grid.compute_kz();
        let kx_s = kx.as_slice().expect("kx contiguous");
        let ky_s = ky.as_slice().expect("ky contiguous");
        let kz_s = kz.as_slice().expect("kz contiguous");

        let mut grad_x_k = LetoArray3::<Complex>::zeros([nx, ny, nz]);
        let mut grad_y_k = LetoArray3::<Complex>::zeros([nx, ny, nz]);
        let mut grad_z_k = LetoArray3::<Complex>::zeros([nx, ny, nz]);
        let mut laplacian_k = LetoArray3::<Complex>::zeros([nx, ny, nz]);

        // Spectral differentiation in one indexed pass over the filtered spectrum.
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let pk = pressure_k[[i, j, k]];
                    let kxi = kx_s[i];
                    let kyj = ky_s[j];
                    let kzk = kz_s[k];
                    let k2 = kxi.mul_add(kxi, kyj.mul_add(kyj, kzk * kzk));
                    grad_x_k[[i, j, k]] = pk * Complex::new(0.0, kxi);
                    grad_y_k[[i, j, k]] = pk * Complex::new(0.0, kyj);
                    grad_z_k[[i, j, k]] = pk * Complex::new(0.0, kzk);
                    laplacian_k[[i, j, k]] = pk * (-k2);
                }
            }
        }

        let p_filt = ndarray_real_field(ifft_3d_array(&pressure_k));
        let grad_x = ndarray_real_field(ifft_3d_array(&grad_x_k));
        let grad_y = ndarray_real_field(ifft_3d_array(&grad_y_k));
        let grad_z = ndarray_real_field(ifft_3d_array(&grad_z_k));
        let laplacian = ndarray_real_field(ifft_3d_array(&laplacian_k));

        let mut nonlinear_term = Array3::zeros((nx, ny, nz));
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;

                    let density = kwavers_medium::density_at(medium, x, y, z, grid);
                    let b_over_a = kwavers_medium::AcousticProperties::nonlinearity_parameter(
                        medium, x, y, z, grid,
                    );

                    // β = 1 + B/(2A) (Def 3.2, SSOT); prefactor 2β·dt²/ρ₀ from Westervelt Ch. 3.
                    let beta = kwavers_medium::properties::coefficient_of_nonlinearity(b_over_a);
                    let prefactor = 2.0 * beta * self.dt.powi(2) / density;

                    let p_lap = p_filt[[i, j, k]] * laplacian[[i, j, k]];
                    let grad_sq = grad_z[[i, j, k]].mul_add(
                        grad_z[[i, j, k]],
                        grad_y[[i, j, k]].mul_add(grad_y[[i, j, k]], grad_x[[i, j, k]].powi(2)),
                    );
                    nonlinear_term[[i, j, k]] = prefactor * (p_lap + grad_sq);
                }
            }
        }

        Ok(nonlinear_term)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::wave_model::NonlinearWave;
    use kwavers_grid::Grid;
    use kwavers_medium::HomogeneousMedium;
    use ndarray::Array3;

    /// A spatially uniform (constant) pressure field has zero gradient and zero
    /// Laplacian. Both terms of the Westervelt nonlinear operator vanish, so the
    /// nonlinear contribution must be identically zero everywhere.
    ///
    /// Tolerance: N·ε_mach·10 where N = nx·ny·nz = 512 for an 8³ grid.
    #[test]
    fn compute_nonlinear_term_zero_for_constant_pressure_field() {
        let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let mut w = NonlinearWave::new(&grid, 1e-7);
        w.precompute_k_squared(&grid);

        // Constant field: gradient = 0, Laplacian = 0 → nonlinear term = 0
        let pressure = Array3::<f64>::from_elem((8, 8, 8), 1_000.0);

        let term = w.compute_nonlinear_term(&pressure, &medium, &grid).unwrap();

        let tol = 512.0 * f64::EPSILON * 10.0;
        for &v in term.iter() {
            assert!(
                v.abs() < tol,
                "nonlinear term must be zero for constant pressure (got {v:.3e}, tol {tol:.3e})"
            );
        }
    }

    /// A pure Nyquist sinusoid along x (index N/2, above the 2/3-rule cutoff N/3)
    /// is zeroed by the dealiasing filter. After filtering, p_filt ≡ 0, so both
    /// p_filt·∇²p_filt and |∇p_filt|² are identically zero.
    ///
    /// Without dealiasing, squaring the Nyquist mode would fold energy onto the DC
    /// bin, producing a spurious constant pressure offset proportional to A². The
    /// filter eliminates this artifact.
    ///
    /// Threshold: max|term| < 1e-6 Pa (FFT round-trip precision on a 12³ grid).
    #[test]
    fn compute_nonlinear_term_suppresses_nyquist_aliasing_via_dealiasing_filter() {
        use std::f64::consts::PI;
        let n = 12usize; // cutoff = n/3 = 4; Nyquist index = n/2 = 6 > 4
        let grid = Grid::new(n, n, n, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let mut w = NonlinearWave::new(&grid, 1e-7);
        w.precompute_k_squared(&grid);

        // cos(π·i) alternates ±1: DFT energy concentrated at index N/2 = 6
        let p_nyquist =
            Array3::from_shape_fn((n, n, n), |(i, _j, _k)| 1e5_f64 * (PI * i as f64).cos());

        let term = w
            .compute_nonlinear_term(&p_nyquist, &medium, &grid)
            .unwrap();

        let max_abs = term.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
        assert!(
            max_abs < 1e-6,
            "dealiased Nyquist mode must give negligible nonlinear term \
             (max |term| = {max_abs:.3e} Pa, expected < 1e-6 Pa)"
        );
    }

    /// Zero pressure field → zero nonlinear term (trivial null case).
    #[test]
    fn compute_nonlinear_term_zero_for_zero_pressure_field() {
        let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let w = NonlinearWave::new(&grid, 1e-7);

        let pressure = Array3::<f64>::zeros((8, 8, 8));
        let term = w.compute_nonlinear_term(&pressure, &medium, &grid).unwrap();

        for &v in term.iter() {
            assert_eq!(v, 0.0, "zero pressure must give zero nonlinear term");
        }
    }
}
