//! CPML profile computation and management.
//!
//! # Physics: Convolutional Perfectly Matched Layer
//!
//! ## Background
//! The CPML extends Berenger's split-field PML with recursive convolution
//! memory variables. For acoustic simulations, CPML modifies gradient
//! operators in absorbing regions by a frequency-dependent complex stretching
//! function in the Laplace domain.
//!
//! ## Theorem (CPML Profile Grading, Roden & Gedney 2000)
//! For PML thickness `d`, polynomial order `m`, target reflection `R0`, and
//! depth `xi in [0,d]` measured from the physical-domain interface:
//!
//! ```text
//! sigma_max = -(m + 1) c0 ln(R0) / (2 d)
//! sigma(xi) = sigma_max (xi / d)^m
//! kappa(xi) = 1 + (kappa_max - 1) (xi / d)^m
//! alpha(xi) = alpha_max (1 - xi / d)
//! ```
//!
//! k-Wave's default acoustic profile fixes `m = 4`, `kappa = 1`, and
//! `alpha = 0`, with `sigma = pml_alpha * (c / dx) * q^4`.
//!
//! ## Theorem (Recursive Convolution Coefficients)
//! With complex-frequency-shift parameters `sigma`, `kappa`, and `alpha`
//! (canonical SEISMIC_CPML / Roden & Gedney 2000 form):
//!
//! ```text
//! b = exp(-(sigma/kappa + alpha) dt)
//! a = sigma (b - 1) / [kappa (sigma + kappa alpha)]   (a = 0 where sigma + kappa alpha = 0)
//! ```
//!
//! The CFS terms are opt-in via [`CPMLConfig::with_cfs_pml`]; the default
//! configuration (`kappa_max = 1`, `alpha_max = 0`) reduces this **exactly** to
//! the σ-only CPML `b = exp(-sigma dt)`, `a = b - 1`. If the memory variable
//! starts at zero, the first corrected gradient is `b * grad`, so the boundary
//! attenuates rather than amplifies the outgoing field. Adding the graded
//! `kappa` (real stretch) and `alpha` (frequency shift) reduces spurious
//! reflections at grazing incidence and for evanescent/low-frequency energy
//! (Komatitsch & Martin 2007/2009); recommended `kappa_max in [5,20]`,
//! `alpha_max ~ pi*f0`.
//!
//! ## References
//! - Roden & Gedney (2000). Microwave Opt. Tech. Lett. 27(5), 334-339.
//!   DOI: 10.1002/1098-2760(20001205)27:5<334::AID-MOP14>3.0.CO;2-A
//! - Collino & Tsogka (2001). Geophysics 66(1), 294-307.
//!   DOI: 10.1190/1.1444908
//! - Berenger (1994). J. Comput. Phys. 114(2), 185-200.
//!   DOI: 10.1006/jcph.1994.1159

mod kernels;
#[cfg(test)]
mod tests;

use super::config::CPMLConfig;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use leto::Array1;

/// CPML absorption and stretching profiles.
///
/// Two sigma profile families are maintained:
/// - `sigma_*`: collocated profiles for split density components.
/// - `sigma_*_sg*`: staggered profiles for velocity components.
///
/// The staggered profiles follow k-Wave's `get_pml(..., staggered=True)`
/// half-cell offset so velocity and density updates use their native grids.
///
/// ## Precomputed PML factors
///
/// `pml_vel_*[i] = exp(-sigma_*_sg*[i] · Δt/2)` and
/// `pml_den_*[i] = exp(-sigma_*[i] · Δt/2)` are computed once at construction
/// from the final sigma arrays (after radial-inner-z transparency is applied).
/// They replace per-step per-element `exp()` evaluation in the split-field PML
/// update, reducing the per-step cost from O(N) transcendental ops to O(N)
/// multiplications.  The fused velocity/density update uses these directly:
/// `u = pml² · u_old − pml · (dt/ρ) · grad_p` becomes
/// `u = p · (p · u_old − (dt/ρ) · grad_p)` with `p = pml_vel_*`.
#[derive(Debug, Clone)]
pub struct CPMLProfiles {
    pub sigma_x: Array1<f64>,
    pub sigma_y: Array1<f64>,
    pub sigma_z: Array1<f64>,
    pub sigma_x_sgx: Array1<f64>,
    pub sigma_y_sgy: Array1<f64>,
    pub sigma_z_sgz: Array1<f64>,
    pub kappa_x: Array1<f64>,
    pub kappa_y: Array1<f64>,
    pub kappa_z: Array1<f64>,
    pub alpha_x: Array1<f64>,
    pub alpha_y: Array1<f64>,
    pub alpha_z: Array1<f64>,
    pub a_x: Array1<f64>,
    pub a_y: Array1<f64>,
    pub a_z: Array1<f64>,
    pub b_x: Array1<f64>,
    pub b_y: Array1<f64>,
    pub b_z: Array1<f64>,
    /// `exp(-sigma_x_sgx[i] · dt/2)` — staggered PML factor for `ux`.
    pub pml_vel_x: Array1<f64>,
    /// `exp(-sigma_y_sgy[j] · dt/2)` — staggered PML factor for `uy`.
    pub pml_vel_y: Array1<f64>,
    /// `exp(-sigma_z_sgz[k] · dt/2)` — staggered PML factor for `uz`.
    pub pml_vel_z: Array1<f64>,
    /// `exp(-sigma_x[i] · dt/2)` — collocated PML factor for `rhox`.
    pub pml_den_x: Array1<f64>,
    /// `exp(-sigma_y[j] · dt/2)` — collocated PML factor for `rhoy`.
    pub pml_den_y: Array1<f64>,
    /// `exp(-sigma_z[k] · dt/2)` — collocated PML factor for `rhoz`.
    pub pml_den_z: Array1<f64>,
}

impl CPMLProfiles {
    /// CPML decay coefficient `b` for `axis` (0=x, 1=y, 2=z).
    #[inline]
    pub(crate) fn b(&self, axis: usize) -> &Array1<f64> {
        match axis {
            0 => &self.b_x,
            1 => &self.b_y,
            _ => &self.b_z,
        }
    }

    /// CPML amplitude coefficient `a` for `axis` (0=x, 1=y, 2=z).
    #[inline]
    pub(crate) fn a(&self, axis: usize) -> &Array1<f64> {
        match axis {
            0 => &self.a_x,
            1 => &self.a_y,
            _ => &self.a_z,
        }
    }

    /// CPML stretch factor `kappa` for `axis` (0=x, 1=y, 2=z).
    #[inline]
    pub(crate) fn kappa(&self, axis: usize) -> &Array1<f64> {
        match axis {
            0 => &self.kappa_x,
            1 => &self.kappa_y,
            _ => &self.kappa_z,
        }
    }

    /// Create CPML profiles from grid spacing, medium reference speed, and time step.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: &CPMLConfig, grid: &Grid, sound_speed: f64, dt: f64) -> KwaversResult<Self> {
        let mut profiles = Self::neutral(grid.nx, grid.ny, grid.nz);
        profiles.compute_profiles(config, grid, sound_speed, dt)?;
        profiles.apply_radial_inner_z_transparency(config, grid.nz);
        // Precompute exp factors AFTER transparency zeroing so the factors
        // reflect the final sigma values (not an intermediate state).
        profiles.compute_exp_factors(dt);
        Ok(profiles)
    }

    fn neutral(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            sigma_x: Array1::zeros([nx]),
            sigma_y: Array1::zeros([ny]),
            sigma_z: Array1::zeros([nz]),
            sigma_x_sgx: Array1::zeros([nx]),
            sigma_y_sgy: Array1::zeros([ny]),
            sigma_z_sgz: Array1::zeros([nz]),
            kappa_x: Array1::ones([nx]),
            kappa_y: Array1::ones([ny]),
            kappa_z: Array1::ones([nz]),
            alpha_x: Array1::zeros([nx]),
            alpha_y: Array1::zeros([ny]),
            alpha_z: Array1::zeros([nz]),
            a_x: Array1::zeros([nx]),
            a_y: Array1::zeros([ny]),
            a_z: Array1::zeros([nz]),
            b_x: Array1::ones([nx]),
            b_y: Array1::ones([ny]),
            b_z: Array1::ones([nz]),
            // Neutral PML: exp(-0 * dt/2) = 1.0 — no attenuation.
            pml_vel_x: Array1::ones([nx]),
            pml_vel_y: Array1::ones([ny]),
            pml_vel_z: Array1::ones([nz]),
            pml_den_x: Array1::ones([nx]),
            pml_den_y: Array1::ones([ny]),
            pml_den_z: Array1::ones([nz]),
        }
    }

    /// Compute `pml_vel_*` and `pml_den_*` from the current sigma arrays.
    ///
    /// Must be called AFTER `apply_radial_inner_z_transparency` so that the
    /// transparency-zeroed sigma values are reflected in the precomputed factors.
    ///
    /// # Theorem (PML factor derivation)
    /// Treeby & Cox (2010) Eq. 17 applies `pml = exp(-σ·Δt/2)` twice per step.
    /// Pre-computing `pml = exp(-σ·Δt/2)` per grid index eliminates O(N)
    /// transcendental evaluations per step, replacing them with O(N) multiplications.
    fn compute_exp_factors(&mut self, dt: f64) {
        self.pml_vel_x = self.sigma_x_sgx.mapv(|s| (-s * dt * 0.5).exp());
        self.pml_vel_y = self.sigma_y_sgy.mapv(|s| (-s * dt * 0.5).exp());
        self.pml_vel_z = self.sigma_z_sgz.mapv(|s| (-s * dt * 0.5).exp());
        self.pml_den_x = self.sigma_x.mapv(|s| (-s * dt * 0.5).exp());
        self.pml_den_y = self.sigma_y.mapv(|s| (-s * dt * 0.5).exp());
        self.pml_den_z = self.sigma_z.mapv(|s| (-s * dt * 0.5).exp());
    }

    fn compute_profiles(
        &mut self,
        config: &CPMLConfig,
        grid: &Grid,
        sound_speed: f64,
        dt: f64,
    ) -> KwaversResult<()> {
        let alpha_x = config.sigma_factor_for_dimension(0)?;
        let alpha_y = config.sigma_factor_for_dimension(1)?;
        let alpha_z = config.sigma_factor_for_dimension(2)?;

        let spec = |dx: f64, thickness: usize, pml_alpha: f64| kernels::CollocatedProfileSpec {
            dx,
            thickness,
            pml_alpha,
            sound_speed,
            dt,
            kappa_max: config.kappa_max,
            alpha_max: config.alpha_max,
        };

        kernels::compute_collocated_profile(
            kernels::CollocatedProfileMut::new(
                &mut self.sigma_x,
                &mut self.kappa_x,
                &mut self.alpha_x,
                &mut self.a_x,
                &mut self.b_x,
            ),
            grid.nx,
            &spec(grid.dx, config.per_dimension.x, alpha_x),
        );
        kernels::compute_collocated_profile(
            kernels::CollocatedProfileMut::new(
                &mut self.sigma_y,
                &mut self.kappa_y,
                &mut self.alpha_y,
                &mut self.a_y,
                &mut self.b_y,
            ),
            grid.ny,
            &spec(grid.dy, config.per_dimension.y, alpha_y),
        );
        kernels::compute_collocated_profile(
            kernels::CollocatedProfileMut::new(
                &mut self.sigma_z,
                &mut self.kappa_z,
                &mut self.alpha_z,
                &mut self.a_z,
                &mut self.b_z,
            ),
            grid.nz,
            &spec(grid.dz, config.per_dimension.z, alpha_z),
        );

        kernels::compute_staggered_profile(
            &mut self.sigma_x_sgx,
            grid.nx,
            grid.dx,
            config.per_dimension.x,
            alpha_x,
            sound_speed,
        );
        kernels::compute_staggered_profile(
            &mut self.sigma_y_sgy,
            grid.ny,
            grid.dy,
            config.per_dimension.y,
            alpha_y,
            sound_speed,
        );
        kernels::compute_staggered_profile(
            &mut self.sigma_z_sgz,
            grid.nz,
            grid.dz,
            config.per_dimension.z,
            alpha_z,
            sound_speed,
        );
        Ok(())
    }

    fn apply_radial_inner_z_transparency(&mut self, config: &CPMLConfig, nz: usize) {
        if !config.radial_inner_z_transparent {
            return;
        }

        for i in 0..config.per_dimension.z.min(nz) {
            self.sigma_z[i] = 0.0;
            self.sigma_z_sgz[i] = 0.0;
            self.kappa_z[i] = 1.0;
            self.alpha_z[i] = 0.0;
            self.a_z[i] = 0.0;
            self.b_z[i] = 1.0;
        }
    }
}
