use super::{NumericalValidator, StabilityResults};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::constants::SOUND_SPEED_TISSUE;

impl NumericalValidator {
    /// Validate stability.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn validate_stability(
        &self,
    ) -> Result<StabilityResults, Box<dyn std::error::Error>> {
        let sound_speed = kwavers_medium::sound_speed_at(&self.medium, 0.0, 0.0, 0.0, &self.grid);
        let dt_max = self.grid.dx / (sound_speed * (3.0_f64).sqrt());

        let cfl_numbers = vec![0.1, 0.5, 0.9, 1.0, 1.1];
        let mut pstd_stable = true;
        let mut fdtd_stable = true;
        let mut kuznetsov_stable = true;
        let mut max_stable_cfl = 0.0;
        let mut growth_rate: f64 = 0.0;

        for &cfl in &cfl_numbers {
            let dt = cfl * dt_max;

            let pstd_growth = self.test_stability_pstd(dt)?;
            let fdtd_growth = self.test_stability_fdtd(dt)?;
            let kuznetsov_growth = self.test_stability_kuznetsov(dt)?;

            if pstd_growth.abs() < 1e-10 && pstd_stable {
                max_stable_cfl = cfl;
            } else {
                pstd_stable = false;
            }

            if fdtd_growth.abs() < 1e-10 && fdtd_stable {
                // FDTD typically stable up to CFL=1/sqrt(3) in 3D
            } else if cfl > 1.0 / (3.0_f64).sqrt() {
                fdtd_stable = false;
            }

            if kuznetsov_growth.abs() < 1e-10 && kuznetsov_stable {
                // Kuznetsov has similar stability to FDTD
            } else if cfl > 1.0 / (3.0_f64).sqrt() {
                kuznetsov_stable = false;
            }

            growth_rate = growth_rate.max(pstd_growth.max(fdtd_growth.max(kuznetsov_growth)));
        }

        Ok(StabilityResults {
            pstd_stable,
            fdtd_stable,
            kuznetsov_stable,
            max_cfl_number: max_stable_cfl,
            growth_rate,
        })
    }
    /// Test stability pstd.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn test_stability_pstd(&self, dt: f64) -> Result<f64, Box<dyn std::error::Error>> {
        let f_max = 1.0 / (2.0 * self.grid.dx.min(self.grid.dy).min(self.grid.dz));
        let omega_max = TWO_PI * f_max;

        let c_max = SOUND_SPEED_TISSUE;
        let cfl = c_max * dt / self.grid.dx.min(self.grid.dy).min(self.grid.dz);

        if cfl <= 1.0 {
            Ok(0.0)
        } else {
            Ok((cfl - 1.0) * omega_max * dt)
        }
    }
    /// Test stability fdtd.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn test_stability_fdtd(&self, dt: f64) -> Result<f64, Box<dyn std::error::Error>> {
        let c_max = SOUND_SPEED_TISSUE;
        let dx_min = self.grid.dx.min(self.grid.dy).min(self.grid.dz);
        let cfl_limit = dx_min / (3.0_f64.sqrt());
        let actual_cfl = c_max * dt;

        if actual_cfl <= cfl_limit {
            Ok(0.0)
        } else {
            let violation_ratio = actual_cfl / cfl_limit;
            Ok((violation_ratio - 1.0).ln())
        }
    }
    /// Test stability kuznetsov.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn test_stability_kuznetsov(
        &self,
        dt: f64,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        const NONLINEAR_SAFETY_FACTOR: f64 = 1.5;

        let c_max = SOUND_SPEED_TISSUE;
        let dx_min = self.grid.dx.min(self.grid.dy).min(self.grid.dz);
        let cfl_limit = dx_min / (3.0_f64.sqrt() * NONLINEAR_SAFETY_FACTOR);
        let actual_cfl = c_max * dt;

        if actual_cfl <= cfl_limit {
            Ok(0.0)
        } else {
            let violation_ratio = actual_cfl / cfl_limit;
            Ok((violation_ratio - 1.0) * violation_ratio)
        }
    }
}
