// src/physics/mechanics/elastic_wave/mod.rs
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::traits::AcousticWaveModel;
use crate::source::Source;
use crate::solver::{VX_IDX, VY_IDX, VZ_IDX, SXX_IDX, SYY_IDX, SZZ_IDX, SXY_IDX, SXZ_IDX, SYZ_IDX};
use crate::utils::{fft_3d, ifft_3d};
use ndarray::{Array3, Array4, Axis, s}; // Removed Zip
use num_complex::Complex;
use log::{debug, trace, warn};
use std::time::Instant; // Added for performance timing
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

        let lambda_arr = medium.lame_lambda_array();
        let mu_arr = medium.lame_mu_array();
        let rho_arr = medium.density_array();

        let fft_start_time = Instant::now();
        let mut temp_field_holder = Array4::<f64>::zeros((1, nx, ny, nz));

        temp_field_holder.slice_mut(s![0, .., .., ..]).assign(&fields.index_axis(Axis(0), VX_IDX));
        let vx_fft = fft_3d(&temp_field_holder, 0, grid);
        temp_field_holder.slice_mut(s![0, .., .., ..]).assign(&fields.index_axis(Axis(0), VY_IDX));
        let vy_fft = fft_3d(&temp_field_holder, 0, grid);
        temp_field_holder.slice_mut(s![0, .., .., ..]).assign(&fields.index_axis(Axis(0), VZ_IDX));
        let vz_fft = fft_3d(&temp_field_holder, 0, grid);

        temp_field_holder.slice_mut(s![0, .., .., ..]).assign(&fields.index_axis(Axis(0), SXX_IDX));
        let sxx_fft = fft_3d(&temp_field_holder, 0, grid);
        temp_field_holder.slice_mut(s![0, .., .., ..]).assign(&fields.index_axis(Axis(0), SYY_IDX));
        let syy_fft = fft_3d(&temp_field_holder, 0, grid);
        temp_field_holder.slice_mut(s![0, .., .., ..]).assign(&fields.index_axis(Axis(0), SZZ_IDX));
        let szz_fft = fft_3d(&temp_field_holder, 0, grid);
        temp_field_holder.slice_mut(s![0, .., .., ..]).assign(&fields.index_axis(Axis(0), SXY_IDX));
        let sxy_fft = fft_3d(&temp_field_holder, 0, grid);
        temp_field_holder.slice_mut(s![0, .., .., ..]).assign(&fields.index_axis(Axis(0), SXZ_IDX));
        let sxz_fft = fft_3d(&temp_field_holder, 0, grid);
        temp_field_holder.slice_mut(s![0, .., .., ..]).assign(&fields.index_axis(Axis(0), SYZ_IDX));
        let syz_fft = fft_3d(&temp_field_holder, 0, grid);
        self.fft_time += fft_start_time.elapsed().as_secs_f64();

        let stress_update_start_time = Instant::now();
        let mut sxx_fft_new = Array3::zeros(sxx_fft.dim());
        let mut syy_fft_new = Array3::zeros(syy_fft.dim());
        let mut szz_fft_new = Array3::zeros(szz_fft.dim());
        let mut sxy_fft_new = Array3::zeros(sxy_fft.dim());
        let mut sxz_fft_new = Array3::zeros(sxz_fft.dim());
        let mut syz_fft_new = Array3::zeros(syz_fft.dim());

        let lambda_avg = lambda_arr.mean().unwrap_or(0.0);
        let mu_avg = mu_arr.mean().unwrap_or(0.0);

        // Sequential loop for stress updates
        // TODO: Revisit parallelization for performance.
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let ikx = Complex::new(0.0, self.kx[[i,j,k]]);
                    let iky = Complex::new(0.0, self.ky[[i,j,k]]);
                    let ikz = Complex::new(0.0, self.kz[[i,j,k]]);

                    let vx_k_val = vx_fft[[i,j,k]];
                    let vy_k_val = vy_fft[[i,j,k]];
                    let vz_k_val = vz_fft[[i,j,k]];

                    let dvx_dx_k = ikx * vx_k_val;
                    let dvy_dy_k = iky * vy_k_val;
                    let dvz_dz_k = ikz * vz_k_val;
                    let dvx_dy_k = iky * vx_k_val;
                    let dvy_dx_k = ikx * vy_k_val;
                    let dvx_dz_k = ikz * vx_k_val;
                    let dvz_dx_k = ikx * vz_k_val;
                    let dvy_dz_k = ikz * vy_k_val;
                    let dvz_dy_k = iky * vz_k_val;

                    sxx_fft_new[[i,j,k]] = sxx_fft[[i,j,k]] + dt * ((lambda_avg + 2.0 * mu_avg) * dvx_dx_k + lambda_avg * (dvy_dy_k + dvz_dz_k));
                    syy_fft_new[[i,j,k]] = syy_fft[[i,j,k]] + dt * ((lambda_avg + 2.0 * mu_avg) * dvy_dy_k + lambda_avg * (dvx_dx_k + dvz_dz_k));
                    szz_fft_new[[i,j,k]] = szz_fft[[i,j,k]] + dt * ((lambda_avg + 2.0 * mu_avg) * dvz_dz_k + lambda_avg * (dvx_dx_k + dvy_dy_k));
                    sxy_fft_new[[i,j,k]] = sxy_fft[[i,j,k]] + dt * mu_avg * (dvx_dy_k + dvy_dx_k);
                    sxz_fft_new[[i,j,k]] = sxz_fft[[i,j,k]] + dt * mu_avg * (dvx_dz_k + dvz_dx_k);
                    syz_fft_new[[i,j,k]] = syz_fft[[i,j,k]] + dt * mu_avg * (dvy_dz_k + dvz_dy_k);
                }
            }
        }
        self.stress_update_time += stress_update_start_time.elapsed().as_secs_f64();

        let ifft_stress_start_time = Instant::now();
        let sxx_new_time = ifft_3d(&sxx_fft_new, grid);
        let syy_new_time = ifft_3d(&syy_fft_new, grid);
        let szz_new_time = ifft_3d(&szz_fft_new, grid);
        let sxy_new_time = ifft_3d(&sxy_fft_new, grid);
        let sxz_new_time = ifft_3d(&sxz_fft_new, grid);
        let syz_new_time = ifft_3d(&syz_fft_new, grid);
        self.fft_time += ifft_stress_start_time.elapsed().as_secs_f64();

        let velocity_update_start_time = Instant::now();
        let mut vx_fft_new = Array3::zeros(vx_fft.dim());
        let mut vy_fft_new = Array3::zeros(vy_fft.dim());
        let mut vz_fft_new = Array3::zeros(vz_fft.dim());

        let source_calc_start_time = Instant::now();
        let mut source_term_spatial_fz = Array3::<f64>::zeros((nx, ny, nz));
        // NOTE: Changed Zip::indexed(...).par_for_each to sequential for source calculation
        // to maintain consistency with the main update loops for now.
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x_coord = i as f64 * grid.dx;
                    let y_coord = j as f64 * grid.dy;
                    let z_coord = k as f64 * grid.dz;
                    source_term_spatial_fz[[i,j,k]] = source.get_source_term(t, x_coord, y_coord, z_coord, grid);
                }
            }
        }
        temp_field_holder.slice_mut(s![0, .., .., ..]).assign(&source_term_spatial_fz);
        let source_term_fz_fft = fft_3d(&temp_field_holder, 0, grid);
        self.source_time += source_calc_start_time.elapsed().as_secs_f64();

        let rho_avg = rho_arr.mean().unwrap_or(1.0).max(1e-3);

        // Sequential update for velocity components
        // TODO: Revisit parallelization for performance.
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let ikx = Complex::new(0.0, self.kx[[i,j,k]]);
                    let iky = Complex::new(0.0, self.ky[[i,j,k]]);
                    let ikz = Complex::new(0.0, self.kz[[i,j,k]]);

                    let sxx_u_val = sxx_fft_new[[i,j,k]]; // Use the NEWLY updated stress FFTs
                    let syy_u_val = syy_fft_new[[i,j,k]];
                    let szz_u_val = szz_fft_new[[i,j,k]];
                    let sxy_u_val = sxy_fft_new[[i,j,k]];
                    let sxz_u_val = sxz_fft_new[[i,j,k]];
                    let syz_u_val = syz_fft_new[[i,j,k]];
                    let fz_k_val = source_term_fz_fft[[i,j,k]];

                    let dsxx_dx_k = ikx * sxx_u_val;
                    let dsyy_dy_k = iky * syy_u_val;
                    let dszz_dz_k = ikz * szz_u_val;
                    let dsxy_dx_k = ikx * sxy_u_val;
                    let dsxy_dy_k = iky * sxy_u_val;
                    let dsxz_dx_k = ikx * sxz_u_val;
                    let dsxz_dz_k = ikz * sxz_u_val;
                    let dsyz_dy_k = iky * syz_u_val;
                    let dsyz_dz_k = ikz * syz_u_val;

                    vx_fft_new[[i,j,k]] = vx_fft[[i,j,k]] + (dt / rho_avg) * (dsxx_dx_k + dsxy_dy_k + dsxz_dz_k);
                    vy_fft_new[[i,j,k]] = vy_fft[[i,j,k]] + (dt / rho_avg) * (dsxy_dx_k + dsyy_dy_k + dsyz_dz_k);
                    vz_fft_new[[i,j,k]] = vz_fft[[i,j,k]] + (dt / rho_avg) * (dsxz_dx_k + dsyz_dy_k + dszz_dz_k + fz_k_val);
                }
            }
        }
        self.velocity_update_time += velocity_update_start_time.elapsed().as_secs_f64();

        let ifft_vel_start_time = Instant::now();
        let vx_new_time = ifft_3d(&vx_fft_new, grid);
        let vy_new_time = ifft_3d(&vy_fft_new, grid);
        let vz_new_time = ifft_3d(&vz_fft_new, grid);
        self.fft_time += ifft_vel_start_time.elapsed().as_secs_f64();

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
        trace!("ElasticWave update for t={:.6e} completed in {:.3e} s", t, overall_start_time.elapsed().as_secs_f64());
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
