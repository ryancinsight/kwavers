//! Aperture-plane phase maps.

use kwavers_core::error::KwaversResult;
use leto::Array2;

use super::model::AberrationCorrection;
use kwavers_core::constants::numerical::TWO_PI;

impl AberrationCorrection<'_> {
    /// Compute the 2D phase aberration map at the aperture plane `z = z_max`.
    ///
    /// Returns `Phi(x,y) = Phi(x,y,z_max)`, the quantity compared against
    /// CT-predicted corrections in adaptive focusing experiments.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn aperture_phase_map(&self, frequency: f64) -> KwaversResult<Array2<f64>> {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;
        let k_water = TWO_PI * frequency / self.c_water;
        let dz = self.grid.dz;
        let mut map = Array2::zeros([nx, ny]);

        for i in 0..nx {
            for j in 0..ny {
                let mut total = 0.0_f64;
                for k in 0..nz {
                    let c_local = self.skull.sound_speed[[i, j, k]];
                    if c_local > 0.0 {
                        let k_local = TWO_PI * frequency / c_local;
                        total += (k_local - k_water) * dz;
                    }
                }
                map[[i, j]] = total;
            }
        }

        Ok(map)
    }
}
