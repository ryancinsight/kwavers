// src/physics/mechanics/elastic_wave/mod.rs
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::traits::AcousticWaveModel;
use crate::source::Source;
use crate::solver::{VX_IDX, VY_IDX, VZ_IDX, SXX_IDX, SYY_IDX, SZZ_IDX, SXY_IDX, SXZ_IDX, SYZ_IDX};
use crate::utils::{fft_3d, ifft_3d};
use ndarray::{Array3, Array4, Axis, s};
use num_complex::Complex;
use log::{debug, trace, warn};
use std::time::Instant;
// Removed PI, was unused

/// Solver for linear isotropic elastic wave propagation using a k-space pseudospectral method.
///
/// This solver updates particle velocities (vx, vy, vz) and stress components
/// (sxx, syy, szz, sxy, sxz, syz) based on the 3D linear elastic wave equations.
///
/// It employs a k-space pseudospectral method, which involves:
/// 1.  Forward FFT of velocity and stress fields.
/// 2.  Calculation of spatial derivatives in k-space (multiplication by `i*k_component`).
/// 3.  Application of the elastic constitutive relations and momentum equations in k-space
///     to update the k-space representations of the fields over a time step `dt`.
/// 4.  Inverse FFT of the updated k-space fields to get the new field values in the spatial domain.
///
/// Current Simplifications/Assumptions:
/// - **Linear, Isotropic Medium:** Assumes linear stress-strain relationship and material properties
///   that are the same in all directions.
/// - **Homogeneous in k-space Step:** For heterogeneous media, properties (Lamé parameters, density)
///   are currently averaged for the k-space update step. This is a simplification and may affect
///   accuracy for highly heterogeneous media.
/// - **Source Term:** The scalar source term from the `Source` trait is interpreted as a body force
///   density in the z-direction (Fz).
/// - **No Viscous Damping (Intrinsic):** The current model does not include intrinsic material damping
///   terms (e.g., from shear or bulk viscosity) beyond numerical dissipation.
/// - **No Explicit PML Interaction:** Boundary conditions, including PMLs, are handled by the main
///   `Solver` and applied to field components. This `ElasticWave` model does not directly
///   implement or interact with PMLs itself. A naive application of acoustic PMLs is currently
///   used as a placeholder by the main solver.
///
/// # Fields
/// - `kx`, `ky`, `kz`: Precomputed k-space vectors for each dimension (as 3D grids).
/// - `k_squared`: Precomputed squared magnitude of k-vectors (`kx^2 + ky^2 + kz^2`).
/// - Performance metrics: `call_count`, `fft_time`, `stress_update_time`, `velocity_update_time`, `source_time`, `total_update_time`.
#[derive(Debug, Clone)]
pub struct ElasticWave {
    kx: Array3<f64>,
    ky: Array3<f64>,
    kz: Array3<f64>,
    // k_squared: Array3<f64>, // Currently unused in this linear model without specific k-dependent corrections
    call_count: usize,
    fft_time: f64,
    stress_update_time: f64,
    velocity_update_time: f64,
    source_time: f64,
    total_update_time: f64,
}

impl ElasticWave {
    /// Creates a new `ElasticWave` solver instance.
    ///
    /// Initializes the solver by precomputing k-space vectors based on the provided `grid`.
    ///
    /// # Arguments
    /// * `grid` - A reference to the `Grid` structure defining the simulation domain and discretization.
    ///
    /// # Returns
    /// A new `ElasticWave` instance.
    pub fn new(grid: &Grid) -> Self {
        let kx_vec = grid.kx();
        let ky_vec = grid.ky();
        let kz_vec = grid.kz();

        let mut kx = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
        let mut ky = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
        let mut kz = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k_idx in 0..grid.nz {
                    kx[[i, j, k_idx]] = kx_vec[i];
                    ky[[i, j, k_idx]] = ky_vec[j];
                    kz[[i, j, k_idx]] = kz_vec[k_idx];
                }
            }
        }

        // let k_squared = grid.k_squared(); // k_squared field is currently unused

        debug!("Initialized ElasticWave solver with 3D k-space grids");
        Self {
            kx,
            ky,
            kz,
            // k_squared, // Field commented out
            call_count: 0,
            fft_time: 0.0,
            stress_update_time: 0.0,
            velocity_update_time: 0.0,
            source_time: 0.0,
            total_update_time: 0.0,
        }
    }
}

impl ElasticWave {
    // ... (existing new method)

    fn _perform_fft(&self, field: &Array3<f64>, grid: &Grid) -> Array3<Complex<f64>> {
        let mut temp_field_holder = Array4::<f64>::zeros((1, grid.nx, grid.ny, grid.nz));
        temp_field_holder.slice_mut(s![0, .., .., ..]).assign(field);
        fft_3d(&temp_field_holder, 0, grid)
    }

    fn _perform_ifft(&self, field_fft: &mut Array3<Complex<f64>>, grid: &Grid) -> Array3<f64> {
        ifft_3d(field_fft, grid)
    }

    fn _update_stress_fft(
        &self,
        vx_fft: &Array3<Complex<f64>>,
        vy_fft: &Array3<Complex<f64>>,
        vz_fft: &Array3<Complex<f64>>,
        sxx_fft: &Array3<Complex<f64>>,
        syy_fft: &Array3<Complex<f64>>,
        szz_fft: &Array3<Complex<f64>>,
        sxy_fft: &Array3<Complex<f64>>,
        sxz_fft: &Array3<Complex<f64>>,
        syz_fft: &Array3<Complex<f64>>,
        medium: &dyn Medium,
        dt: f64,
    ) -> (
        Array3<Complex<f64>>,
        Array3<Complex<f64>>,
        Array3<Complex<f64>>,
        Array3<Complex<f64>>,
        Array3<Complex<f64>>,
        Array3<Complex<f64>>,
    ) {
        let (nx, ny, nz) = sxx_fft.dim();
        let mut sxx_fft_new = Array3::zeros((nx, ny, nz));
        let mut syy_fft_new = Array3::zeros((nx, ny, nz));
        let mut szz_fft_new = Array3::zeros((nx, ny, nz));
        let mut sxy_fft_new = Array3::zeros((nx, ny, nz));
        let mut sxz_fft_new = Array3::zeros((nx, ny, nz));
        let mut syz_fft_new = Array3::zeros((nx, ny, nz));

        let lambda = medium.lame_lambda_array();
        let mu = medium.lame_mu_array();
        let shear_viscosity = medium.shear_viscosity_coeff_array();
        let bulk_viscosity = medium.bulk_viscosity_coeff_array();

        for i in 0..sxx_fft.dim().0 {
            for j in 0..sxx_fft.dim().1 {
                for k in 0..sxx_fft.dim().2 {
                    let lambda_val = lambda[[i, j, k]];
            let mu_val = mu[[i, j, k]];

            let ikx = Complex::new(0.0, self.kx[[i, j, k]]);
            let iky = Complex::new(0.0, self.ky[[i, j, k]]);
            let ikz = Complex::new(0.0, self.kz[[i, j, k]]);

            let vx_k_val = vx_fft[[i, j, k]];
            let vy_k_val = vy_fft[[i, j, k]];
            let vz_k_val = vz_fft[[i, j, k]];

            let dvx_dx_k = ikx * vx_k_val;
            let dvy_dy_k = iky * vy_k_val;
            let dvz_dz_k = ikz * vz_k_val;
            let dvx_dy_k = iky * vx_k_val;
            let dvy_dx_k = ikx * vy_k_val;
            let dvx_dz_k = ikz * vx_k_val;
            let dvz_dx_k = ikx * vz_k_val;
            let dvy_dz_k = ikz * vy_k_val;
            let dvz_dy_k = iky * vz_k_val;

            let divergence_v_k = dvx_dx_k + dvy_dy_k + dvz_dz_k;
            let shear_viscosity_val = shear_viscosity[[i, j, k]];
            let bulk_viscosity_val = bulk_viscosity[[i, j, k]];

            let viscous_term_sxx = bulk_viscosity_val * divergence_v_k + 2.0 * shear_viscosity_val * (dvx_dx_k - (1.0/3.0) * divergence_v_k);
            let viscous_term_syy = bulk_viscosity_val * divergence_v_k + 2.0 * shear_viscosity_val * (dvy_dy_k - (1.0/3.0) * divergence_v_k);
            let viscous_term_szz = bulk_viscosity_val * divergence_v_k + 2.0 * shear_viscosity_val * (dvz_dz_k - (1.0/3.0) * divergence_v_k);
            let viscous_term_sxy = shear_viscosity_val * (dvx_dy_k + dvy_dx_k);
            let viscous_term_sxz = shear_viscosity_val * (dvx_dz_k + dvz_dx_k);
            let viscous_term_syz = shear_viscosity_val * (dvy_dz_k + dvz_dy_k);


            sxx_fft_new[[i, j, k]] = sxx_fft[[i, j, k]]
                + dt * (((lambda_val + 2.0 * mu_val) * dvx_dx_k + lambda_val * (dvy_dy_k + dvz_dz_k)) - viscous_term_sxx);
            syy_fft_new[[i, j, k]] = syy_fft[[i, j, k]]
                + dt * (((lambda_val + 2.0 * mu_val) * dvy_dy_k + lambda_val * (dvx_dx_k + dvz_dz_k)) - viscous_term_syy);
            szz_fft_new[[i, j, k]] = szz_fft[[i, j, k]]
                + dt * (((lambda_val + 2.0 * mu_val) * dvz_dz_k + lambda_val * (dvx_dx_k + dvy_dy_k)) - viscous_term_szz);
            sxy_fft_new[[i, j, k]] = sxy_fft[[i, j, k]] + dt * (mu_val * (dvx_dy_k + dvy_dx_k) - viscous_term_sxy);
            sxz_fft_new[[i, j, k]] = sxz_fft[[i, j, k]] + dt * (mu_val * (dvx_dz_k + dvz_dx_k) - viscous_term_sxz);
            syz_fft_new[[i, j, k]] = syz_fft[[i, j, k]] + dt * (mu_val * (dvy_dz_k + dvz_dy_k) - viscous_term_syz);
                }
            }
        }

        (
            sxx_fft_new,
            syy_fft_new,
            szz_fft_new,
            sxy_fft_new,
            sxz_fft_new,
            syz_fft_new,
        )
    }

    fn _update_velocity_fft(
        &self,
        vx_fft: &Array3<Complex<f64>>,
        vy_fft: &Array3<Complex<f64>>,
        vz_fft: &Array3<Complex<f64>>,
        sxx_fft_new: &Array3<Complex<f64>>,
        syy_fft_new: &Array3<Complex<f64>>,
        szz_fft_new: &Array3<Complex<f64>>,
        sxy_fft_new: &Array3<Complex<f64>>,
        sxz_fft_new: &Array3<Complex<f64>>,
        syz_fft_new: &Array3<Complex<f64>>,
        source_term_fz_fft: &Array3<Complex<f64>>,
        medium: &dyn Medium,
        dt: f64,
    ) -> (
        Array3<Complex<f64>>,
        Array3<Complex<f64>>,
        Array3<Complex<f64>>,
    ) {
        let (nx, ny, nz) = vx_fft.dim();
        let mut vx_fft_new = Array3::zeros((nx, ny, nz));
        let mut vy_fft_new = Array3::zeros((nx, ny, nz));
        let mut vz_fft_new = Array3::zeros((nx, ny, nz));

        let rho = medium.density_array();

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let ikx = Complex::new(0.0, self.kx[[i, j, k]]);
            let iky = Complex::new(0.0, self.ky[[i, j, k]]);
            let ikz = Complex::new(0.0, self.kz[[i, j, k]]);

            let sxx_u_val = sxx_fft_new[[i, j, k]];
            let syy_u_val = syy_fft_new[[i, j, k]];
            let szz_u_val = szz_fft_new[[i, j, k]];
            let sxy_u_val = sxy_fft_new[[i, j, k]];
            let sxz_u_val = sxz_fft_new[[i, j, k]];
            let syz_u_val = syz_fft_new[[i, j, k]];
            let fz_k_val = source_term_fz_fft[[i, j, k]];

            let dsxx_dx_k = ikx * sxx_u_val;
            let dsyy_dy_k = iky * syy_u_val;
            let dszz_dz_k = ikz * szz_u_val;
            let dsxy_dx_k = ikx * sxy_u_val;
            let dsxy_dy_k = iky * sxy_u_val;
            let dsxz_dx_k = ikx * sxz_u_val;
            let dsxz_dz_k = ikz * sxz_u_val;
            let dsyz_dy_k = iky * syz_u_val;
            let dsyz_dz_k = ikz * syz_u_val;

            let rho_val = rho[[i, j, k]].max(1e-3);
            vx_fft_new[[i, j, k]] =
                vx_fft[[i, j, k]] + (dt / rho_val) * (dsxx_dx_k + dsxy_dy_k + dsxz_dz_k);
            vy_fft_new[[i, j, k]] =
                vy_fft[[i, j, k]] + (dt / rho_val) * (dsxy_dx_k + dsyy_dy_k + dsyz_dz_k);
            vz_fft_new[[i, j, k]] = vz_fft[[i, j, k]]
                + (dt / rho_val) * (dsxz_dx_k + dsyz_dy_k + dszz_dz_k + fz_k_val);
                }
            }
        }

        (vx_fft_new, vy_fft_new, vz_fft_new)
    }

    fn _apply_source_term(
        &self,
        source: &dyn Source,
        grid: &Grid,
        t: f64,
    ) -> Array3<Complex<f64>> {
        let mut source_term_spatial_fz = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let x_coord = i as f64 * grid.dx;
                    let y_coord = j as f64 * grid.dy;
                    let z_coord = k as f64 * grid.dz;
                    source_term_spatial_fz[[i, j, k]] =
                        source.get_source_term(t, x_coord, y_coord, z_coord, grid);
                }
            }
        }
        self._perform_fft(&source_term_spatial_fz, grid)
    }
}

impl AcousticWaveModel for ElasticWave {
    fn update_wave(
        &mut self,
        fields: &mut Array4<f64>,
        _prev_pressure: &Array3<f64>,
        source: &dyn Source,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) {
        let overall_start_time = Instant::now();
        self.call_count += 1;
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

        if nx == 0 || ny == 0 || nz == 0 {
            trace!("ElasticWave update skipped for empty grid at t={}", t);
            return;
        }

        // FFT of all fields
        let fft_start_time = Instant::now();
        let vx_fft = self._perform_fft(&fields.index_axis(Axis(0), VX_IDX).to_owned(), grid);
        let vy_fft = self._perform_fft(&fields.index_axis(Axis(0), VY_IDX).to_owned(), grid);
        let vz_fft = self._perform_fft(&fields.index_axis(Axis(0), VZ_IDX).to_owned(), grid);
        let sxx_fft = self._perform_fft(&fields.index_axis(Axis(0), SXX_IDX).to_owned(), grid);
        let syy_fft = self._perform_fft(&fields.index_axis(Axis(0), SYY_IDX).to_owned(), grid);
        let szz_fft = self._perform_fft(&fields.index_axis(Axis(0), SZZ_IDX).to_owned(), grid);
        let sxy_fft = self._perform_fft(&fields.index_axis(Axis(0), SXY_IDX).to_owned(), grid);
        let sxz_fft = self._perform_fft(&fields.index_axis(Axis(0), SXZ_IDX).to_owned(), grid);
        let syz_fft = self._perform_fft(&fields.index_axis(Axis(0), SYZ_IDX).to_owned(), grid);
        self.fft_time += fft_start_time.elapsed().as_secs_f64();

        // Update stress fields in k-space
        let stress_update_start_time = Instant::now();
        let (mut sxx_fft_new, mut syy_fft_new, mut szz_fft_new, mut sxy_fft_new, mut sxz_fft_new, mut syz_fft_new) =
            self._update_stress_fft(
                &vx_fft, &vy_fft, &vz_fft, &sxx_fft, &syy_fft, &szz_fft, &sxy_fft, &sxz_fft,
                &syz_fft, medium, dt,
            );
        self.stress_update_time += stress_update_start_time.elapsed().as_secs_f64();

        // Calculate source term in k-space
        let source_calc_start_time = Instant::now();
        let source_term_fz_fft = self._apply_source_term(source, grid, t);
        self.source_time += source_calc_start_time.elapsed().as_secs_f64();

        // Update velocity fields in k-space
        let velocity_update_start_time = Instant::now();
        let (mut vx_fft_new, mut vy_fft_new, mut vz_fft_new) = self._update_velocity_fft(
            &vx_fft,
            &vy_fft,
            &vz_fft,
            &sxx_fft_new,
            &syy_fft_new,
            &szz_fft_new,
            &sxy_fft_new,
            &sxz_fft_new,
            &syz_fft_new,
            &source_term_fz_fft,
            medium,
            dt,
        );
        self.velocity_update_time += velocity_update_start_time.elapsed().as_secs_f64();

        // IFFT of all updated fields
        let ifft_start_time = Instant::now();
        let vx_new_time = self._perform_ifft(&mut vx_fft_new, grid);
        let vy_new_time = self._perform_ifft(&mut vy_fft_new, grid);
        let vz_new_time = self._perform_ifft(&mut vz_fft_new, grid);
        let sxx_new_time = self._perform_ifft(&mut sxx_fft_new, grid);
        let syy_new_time = self._perform_ifft(&mut syy_fft_new, grid);
        let szz_new_time = self._perform_ifft(&mut szz_fft_new, grid);
        let sxy_new_time = self._perform_ifft(&mut sxy_fft_new, grid);
        let sxz_new_time = self._perform_ifft(&mut sxz_fft_new, grid);
        let syz_new_time = self._perform_ifft(&mut syz_fft_new, grid);
        self.fft_time += ifft_start_time.elapsed().as_secs_f64();

        // Assign updated fields back to the main array
        fields.index_axis_mut(Axis(0), VX_IDX).assign(&vx_new_time);
        fields.index_axis_mut(Axis(0), VY_IDX).assign(&vy_new_time);
        fields.index_axis_mut(Axis(0), VZ_IDX).assign(&vz_new_time);
        fields.index_axis_mut(Axis(0), SXX_IDX).assign(&sxx_new_time);
        fields.index_axis_mut(Axis(0), SYY_IDX).assign(&syy_new_time);
        fields.index_axis_mut(Axis(0), SZZ_IDX).assign(&szz_new_time);
        fields.index_axis_mut(Axis(0), SXY_IDX).assign(&sxy_new_time);
        fields.index_axis_mut(Axis(0), SXZ_IDX).assign(&sxz_new_time);
        fields.index_axis_mut(Axis(0), SYZ_IDX).assign(&syz_new_time);

        self.total_update_time += overall_start_time.elapsed().as_secs_f64();
        trace!(
            "ElasticWave update for t={:.6e} completed in {:.3e} s",
            t,
            overall_start_time.elapsed().as_secs_f64()
        );
    }

    fn report_performance(&self) {
        if self.call_count == 0 {
            debug!("No calls to ElasticWave::update_wave yet.");
            return;
        }
        let avg_total_time = self.total_update_time / self.call_count as f64;
        debug!("ElasticWave Performance Report ({} calls):", self.call_count);
        debug!("  Avg Total Update Time: {:.3e} s", avg_total_time);
        if self.total_update_time > 0.0 {
            debug!("    FFT Time:           {:.3e} s ({:.1}%)", self.fft_time / self.call_count as f64, 100.0 * self.fft_time / self.total_update_time);
            debug!("    Stress Update Time: {:.3e} s ({:.1}%)", self.stress_update_time / self.call_count as f64, 100.0 * self.stress_update_time / self.total_update_time);
            debug!("    Velocity Update Time: {:.3e} s ({:.1}%)", self.velocity_update_time / self.call_count as f64, 100.0 * self.velocity_update_time / self.total_update_time);
            debug!("    Source Calc Time:   {:.3e} s ({:.1}%)", self.source_time / self.call_count as f64, 100.0 * self.source_time / self.total_update_time);
        }
    }

    fn set_nonlinearity_scaling(&mut self, _scaling: f64) {
        warn!("Nonlinearity scaling is not applicable to the current Linear ElasticWave model.");
    }

    fn set_k_space_correction_order(&mut self, _order: usize) {
        warn!("K-space correction order is not applicable to the current Linear ElasticWave model in the same way as NonlinearWave.");
    }
}

// TODO: Add unit tests for ElasticWave
// - Test constructor
// - Test a single step update with a known simple scenario (e.g., 1D wave propagation if possible to simplify)
// - Test energy conservation (for a lossless medium) or decay (for a viscous medium if implemented)
// - Test against analytical solutions if available for simple cases.

// TODO: Heterogeneity Handling:
// The current k-space implementation averages medium properties (lambda, mu, rho).
// This is a significant simplification for heterogeneous media.
// More advanced k-space methods (e.g., k-space interaction method, or split-step methods
// where parts are done in spatial domain) are needed for accurate heterogeneous elastic modeling.
// This will be a future improvement.

// TODO: Viscous Damping:
// The equations currently do not include viscous damping terms (using shear and bulk viscosity).
// These would typically be added in k-space similar to how viscosity is handled in NonlinearWave:
// e.g., by multiplying k-space components by exp(-eta * k^2 * dt / rho).
// For elastic waves, it's more complex:
// Stress rate terms get: mu_s * (d/dt Del v_i) + (mu_b - 2/3 mu_s) * (d/dt Del . v) delta_ij
// where mu_s is shear viscosity, mu_b is bulk viscosity.
// This will also be a future improvement.

// TODO: PML for Elastic Waves:
// The current model does not interact with PMLs yet. A specialized PML for elastic waves
// (e.g., CPML) is needed and is significantly more complex than scalar PMLs.
// This is a major future improvement.

#[cfg(test)]
mod tests;
