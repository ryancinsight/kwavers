use crate::core::error::KwaversResult;
use ndarray::Zip;
use num_complex::Complex64;
use std::f64::consts::PI;

use super::ModifiedBornSolver;

impl ModifiedBornSolver {
    /// Apply viscoacoustic green.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn apply_viscoacoustic_green(
        &mut self,
        wavenumber: f64,
        _frequency: f64,
    ) -> KwaversResult<()> {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;

        let heterogeneity = self.workspace.heterogeneity_workspace.view();
        let absorption_field = self.absorption_field.view();

        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let dz = self.grid.dz;

        Zip::indexed(&mut self.workspace.green_workspace).par_for_each(|(i, j, k), green_val| {
            let mut sum = Complex64::new(0.0, 0.0);

            {
                let source_val = heterogeneity[[i, j, k]];
                let absorption = absorption_field[[i, j, k]];
                let k_complex = Complex64::new(wavenumber, absorption.im);
                let self_green = Complex64::new(0.5, 0.0) / k_complex.norm_sqr();
                sum += self_green * source_val;
            }

            let neighbors = [
                (i.saturating_sub(1), j, k),
                ((i + 1).min(nx - 1), j, k),
                (i, j.saturating_sub(1), k),
                (i, (j + 1).min(ny - 1), k),
                (i, j, k.saturating_sub(1)),
                (i, j, (k + 1).min(nz - 1)),
            ];

            for (ni, nj, nk) in neighbors {
                if ni == i && nj == j && nk == k {
                    continue;
                }

                let dist_x = (i as f64 - ni as f64) * dx;
                let dist_y = (j as f64 - nj as f64) * dy;
                let dist_z = (k as f64 - nk as f64) * dz;
                let r = dist_z
                    .mul_add(dist_z, dist_x.mul_add(dist_x, dist_y * dist_y))
                    .sqrt();

                if r > 1e-12 {
                    let source_val = heterogeneity[[ni, nj, nk]];
                    let absorption = absorption_field[[ni, nj, nk]];

                    let kr_real = wavenumber * r;
                    let kr_imag = absorption.im * r;
                    let exp_factor = Complex64::from_polar(1.0, kr_real)
                        * Complex64::exp(Complex64::new(0.0, -kr_imag));
                    let g_val = exp_factor / (4.0 * PI * r);

                    sum += g_val * source_val;
                }
            }

            *green_val = sum;
        });

        Ok(())
    }
}
