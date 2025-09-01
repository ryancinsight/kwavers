//! CPML profile computation and management

use super::config::CPMLConfig;
use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::Array1;

/// CPML absorption and stretching profiles
#[derive(Debug, Clone)]
pub struct CPMLProfiles {
    pub sigma_x: Array1<f64>,
    pub sigma_y: Array1<f64>,
    pub sigma_z: Array1<f64>,
    pub kappa_x: Array1<f64>,
    pub kappa_y: Array1<f64>,
    pub kappa_z: Array1<f64>,
    pub alpha_x: Array1<f64>,
    pub alpha_y: Array1<f64>,
    pub alpha_z: Array1<f64>,
}

impl CPMLProfiles {
    /// Create new CPML profiles based on configuration
    pub fn new(config: &CPMLConfig, grid: &Grid, sound_speed: f64) -> KwaversResult<Self> {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;

        // Initialize profile arrays
        let mut profiles = Self {
            sigma_x: Array1::zeros(nx),
            sigma_y: Array1::zeros(ny),
            sigma_z: Array1::zeros(nz),
            kappa_x: Array1::ones(nx),
            kappa_y: Array1::ones(ny),
            kappa_z: Array1::ones(nz),
            alpha_x: Array1::zeros(nx),
            alpha_y: Array1::zeros(ny),
            alpha_z: Array1::zeros(nz),
        };

        profiles.compute_profiles(config, grid, sound_speed)?;
        Ok(profiles)
    }

    fn compute_profiles(
        &mut self,
        config: &CPMLConfig,
        grid: &Grid,
        sound_speed: f64,
    ) -> KwaversResult<()> {
        // Compute profiles for each direction
        Self::compute_profile_1d(
            &mut self.sigma_x,
            &mut self.kappa_x,
            &mut self.alpha_x,
            grid.nx,
            grid.dx,
            config,
            sound_speed,
        );
        Self::compute_profile_1d(
            &mut self.sigma_y,
            &mut self.kappa_y,
            &mut self.alpha_y,
            grid.ny,
            grid.dy,
            config,
            sound_speed,
        );
        Self::compute_profile_1d(
            &mut self.sigma_z,
            &mut self.kappa_z,
            &mut self.alpha_z,
            grid.nz,
            grid.dz,
            config,
            sound_speed,
        );
        Ok(())
    }

    fn compute_profile_1d(
        sigma: &mut Array1<f64>,
        kappa: &mut Array1<f64>,
        alpha: &mut Array1<f64>,
        n: usize,
        dx: f64,
        config: &CPMLConfig,
        _sound_speed: f64,
    ) {
        let thickness = config.thickness;
        let m = config.polynomial_order;

        // Compute maximum conductivity
        let sigma_max = config.sigma_factor * (m + 1.0) / (150.0 * std::f64::consts::PI * dx);

        // Left boundary
        for i in 0..thickness {
            let d = (thickness - i) as f64 / thickness as f64;
            let d_m = d.powf(m);

            sigma[i] = sigma_max * d_m;
            kappa[i] = 1.0 + (config.kappa_max - 1.0) * d_m;
            alpha[i] = config.alpha_max * (1.0 - d);
        }

        // Right boundary
        for i in (n - thickness)..n {
            let d = (i - (n - thickness) + 1) as f64 / thickness as f64;
            let d_m = d.powf(m);

            sigma[i] = sigma_max * d_m;
            kappa[i] = 1.0 + (config.kappa_max - 1.0) * d_m;
            alpha[i] = config.alpha_max * (1.0 - d);
        }
    }
}
