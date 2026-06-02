//! Volumetric phase-screen integration.

use kwavers_core::error::KwaversResult;
use ndarray::Array3;

use super::model::AberrationCorrection;
use kwavers_core::constants::numerical::TWO_PI;

impl AberrationCorrection<'_> {
    /// Compute the volumetric running phase integral `Phi(x,y,z)`.
    ///
    /// # Formula
    ///
    /// `Phi(i,j,k) = sum_{k'=0}^k [k_local(i,j,k') - k_water] dz`, with
    /// `k_local = 2 pi f / c_local` and `k_water = 2 pi f / c_water`.
    ///
    /// Positive values indicate a slower-than-water path; cortical bone
    /// normally gives negative values because `c_bone > c_water`.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn compute_time_reversal_phases(&self, frequency: f64) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        let dz = self.grid.dz;
        let k_water = TWO_PI * frequency / self.c_water;
        let mut phases = Array3::zeros((nx, ny, nz));

        for i in 0..nx {
            for j in 0..ny {
                let mut running_phase = 0.0_f64;
                for k in 0..nz {
                    let c_local = self.skull.sound_speed[[i, j, k]];
                    if c_local > 0.0 {
                        let k_local = TWO_PI * frequency / c_local;
                        running_phase += (k_local - k_water) * dz;
                    }
                    phases[[i, j, k]] = running_phase;
                }
            }
        }

        Ok(phases)
    }

    /// Compute the correction phase field `Phi_corr = -Phi(x,y,z)`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn compute_correction_phases(&self, frequency: f64) -> KwaversResult<Array3<f64>> {
        let phases = self.compute_time_reversal_phases(frequency)?;
        Ok(phases.mapv(|phi| -phi))
    }
}
