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
//! With complex-frequency-shift parameters `sigma`, `kappa`, and `alpha`:
//!
//! ```text
//! b = exp(-(sigma/kappa + alpha) dt)
//! a = (sigma/kappa) (b - 1) / (sigma/kappa + alpha)
//! ```
//!
//! The specialization used here has `alpha = 0` and `kappa = 1`, therefore
//! `b = exp(-sigma dt)` and `a = b - 1`. If the memory variable starts at
//! zero, the first corrected gradient is `b * grad`, so the boundary attenuates
//! rather than amplifies the outgoing field.
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
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::Array1;

/// CPML absorption and stretching profiles.
///
/// Two sigma profile families are maintained:
/// - `sigma_*`: collocated profiles for split density components.
/// - `sigma_*_sg*`: staggered profiles for velocity components.
///
/// The staggered profiles follow k-Wave's `get_pml(..., staggered=True)`
/// half-cell offset so velocity and density updates use their native grids.
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
}

impl CPMLProfiles {
    /// Create CPML profiles from grid spacing, medium reference speed, and time step.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: &CPMLConfig, grid: &Grid, sound_speed: f64, dt: f64) -> KwaversResult<Self> {
        let mut profiles = Self::neutral(grid.nx, grid.ny, grid.nz);
        profiles.compute_profiles(config, grid, sound_speed, dt)?;
        profiles.apply_radial_inner_z_transparency(config, grid.nz);
        Ok(profiles)
    }

    fn neutral(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            sigma_x: Array1::zeros(nx),
            sigma_y: Array1::zeros(ny),
            sigma_z: Array1::zeros(nz),
            sigma_x_sgx: Array1::zeros(nx),
            sigma_y_sgy: Array1::zeros(ny),
            sigma_z_sgz: Array1::zeros(nz),
            kappa_x: Array1::ones(nx),
            kappa_y: Array1::ones(ny),
            kappa_z: Array1::ones(nz),
            alpha_x: Array1::zeros(nx),
            alpha_y: Array1::zeros(ny),
            alpha_z: Array1::zeros(nz),
            a_x: Array1::zeros(nx),
            a_y: Array1::zeros(ny),
            a_z: Array1::zeros(nz),
            b_x: Array1::ones(nx),
            b_y: Array1::ones(ny),
            b_z: Array1::ones(nz),
        }
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

        kernels::compute_collocated_profile(
            kernels::CollocatedProfileMut::new(
                &mut self.sigma_x,
                &mut self.kappa_x,
                &mut self.alpha_x,
                &mut self.a_x,
                &mut self.b_x,
            ),
            grid.nx,
            grid.dx,
            config.per_dimension.x,
            alpha_x,
            sound_speed,
            dt,
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
            grid.dy,
            config.per_dimension.y,
            alpha_y,
            sound_speed,
            dt,
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
            grid.dz,
            config.per_dimension.z,
            alpha_z,
            sound_speed,
            dt,
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
