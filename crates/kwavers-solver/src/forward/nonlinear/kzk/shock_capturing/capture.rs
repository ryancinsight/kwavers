//! Artificial viscosity and shock filter application.

use super::{ShockCapture, ShockDetectionResult};
use kwavers_core::error::KwaversResult;
use leto::Array2;

impl ShockCapture {
    /// Compute artificial viscosity source term: `Q_av = μ |∇p| ∇²p / ρ₀`
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn artificial_viscosity(
        &self,
        pressure: &Array2<f64>,
        dx: f64,
        dz: f64,
        rho0: f64,
        c0: f64,
    ) -> KwaversResult<Array2<f64>> {
        let (nx, nz) = pressure.dim();
        let mut q_av = Array2::zeros((nx, nz));

        if !self.config.enable_capturing || nx < 3 || nz < 3 {
            return Ok(q_av);
        }

        let mut laplacian = Array2::zeros((nx - 2, nz - 2));

        for z in 1..nz - 1 {
            for x in 1..nx - 1 {
                let laplacian_val = (2.0f64.mul_add(-pressure[[x, z]], pressure[[x + 1, z]])
                    + pressure[[x - 1, z]])
                    / (dx * dx)
                    + (2.0f64.mul_add(-pressure[[x, z]], pressure[[x, z + 1]])
                        + pressure[[x, z - 1]])
                        / (dz * dz);
                laplacian[[x - 1, z - 1]] = laplacian_val;
            }
        }

        let mut grad_mag = Array2::zeros((nx - 2, nz - 2));

        for z in 1..nz - 1 {
            for x in 1..nx - 1 {
                let gx = (pressure[[x + 1, z]] - pressure[[x - 1, z]]) / (2.0 * dx);
                let gz = (pressure[[x, z + 1]] - pressure[[x, z - 1]]) / (2.0 * dz);
                grad_mag[[x - 1, z - 1]] = gx.hypot(gz);
            }
        }

        let mu = self.config.viscosity_coefficient * c0;

        for z in 1..nz - 1 {
            for x in 1..nx - 1 {
                q_av[[x, z]] = mu * grad_mag[[x - 1, z - 1]] * laplacian[[x - 1, z - 1]] / rho0;
            }
        }

        Ok(q_av)
    }

    /// Apply shock capturing filter to smooth discontinuities near the shock location.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn shock_filter(
        &self,
        pressure: &mut Array2<f64>,
        shock_result: &ShockDetectionResult,
        filter_width: usize,
    ) -> KwaversResult<()> {
        if !shock_result.shock_detected {
            return Ok(());
        }

        let (nx, nz) = pressure.dim();

        if let Some(z_shock) = shock_result.shock_location {
            let z_min = z_shock.saturating_sub(filter_width);
            let z_max = (z_shock + filter_width).min(nz - 1);

            for z in z_min..=z_max {
                if z > 0 && z < nz - 1 {
                    for x in 1..nx - 1 {
                        let smoothed = (2.0f64.mul_add(pressure[[x, z]], pressure[[x, z - 1]])
                            + pressure[[x, z + 1]])
                            / 4.0;
                        pressure[[x, z]] = smoothed;
                    }
                }
            }
        }

        Ok(())
    }
}
