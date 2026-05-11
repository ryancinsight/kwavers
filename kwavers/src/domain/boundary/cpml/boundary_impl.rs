//! `Boundary` trait impl for `CPMLBoundary`

use super::CPMLBoundary;
use crate::core::error::KwaversResult;
use crate::domain::boundary::Boundary;
use crate::domain::grid::Grid;
use ndarray::{Array3, ArrayViewMut3, Zip};

impl Boundary for CPMLBoundary {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn apply_acoustic(
        &mut self,
        mut field: ArrayViewMut3<f64>,
        grid: &Grid,
        _time_step: usize,
    ) -> KwaversResult<()> {
        // K-Wave applies PML as exp(-sigma * dt/2) to velocity AND density fields
        // (split-field PML, applied twice per step = net exp(-sigma*dt)).
        let dt = self.estimate_dt_from_grid(grid);

        Zip::indexed(&mut field).par_for_each(|(i, j, k), val| {
            let s_x = self.profiles.sigma_x[i];
            let s_y = self.profiles.sigma_y[j];
            let s_z = self.profiles.sigma_z[k];
            let sigma_total = s_x + s_y + s_z;

            if sigma_total > 0.0 {
                *val *= (-sigma_total * dt * 0.5).exp();
            }
        });

        Ok(())
    }

    fn apply_acoustic_freq(
        &mut self,
        field: &mut Array3<num_complex::Complex<f64>>,
        grid: &Grid,
        _time_step: usize,
    ) -> KwaversResult<()> {
        let dt = self.estimate_dt_from_grid(grid);

        Zip::indexed(field).par_for_each(|(i, j, k), val| {
            let s_x = self.profiles.sigma_x[i];
            let s_y = self.profiles.sigma_y[j];
            let s_z = self.profiles.sigma_z[k];
            let sigma_total = s_x + s_y + s_z;

            if sigma_total > 0.0 {
                let decay = (-sigma_total * dt * 0.5).exp();
                val.re *= decay;
                val.im *= decay;
            }
        });

        Ok(())
    }

    /// Apply directional (split-field) PML to a single field component.
    ///
    /// Applies `exp(-sigma_d * dt/2)` where `sigma_d` is the PML profile for
    /// dimension `axis` (0=x, 1=y, 2=z). This matches k-Wave's split-field PML:
    ///   rho_x *= pml_x,  rho_y *= pml_y,  rho_z *= pml_z
    ///   ux    *= pml_x,  uy    *= pml_y,  uz    *= pml_z
    ///
    /// Ref: Treeby & Cox (2010), J. Biomed. Opt. 15(2), Eq. (3)-(5)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn apply_acoustic_directional(
        &mut self,
        mut field: ArrayViewMut3<f64>,
        grid: &Grid,
        _time_step: usize,
        axis: usize,
    ) -> KwaversResult<()> {
        let dt = self.estimate_dt_from_grid(grid);
        Zip::indexed(&mut field).par_for_each(|(i, j, k), val| {
            let sigma = match axis {
                0 => self.profiles.sigma_x[i],
                1 => self.profiles.sigma_y[j],
                _ => self.profiles.sigma_z[k],
            };
            if sigma > 0.0 {
                *val *= (-sigma * dt * 0.5).exp();
            }
        });
        Ok(())
    }

    /// Apply staggered-grid PML to a velocity component.
    ///
    /// Uses `sigma_x_sgx`, `sigma_y_sgy`, or `sigma_z_sgz` (half-cell-shifted sigma)
    /// matching k-Wave's `pml_x_sgx`, `pml_y_sgy`, `pml_z_sgz` arrays.
    ///
    /// At the deepest left PML cell (index 0), the staggered sigma is:
    ///   σ_sg = σ_max · ((pml_size − 0.5) / pml_size)⁴ ≈ 0.706 · σ_max
    /// vs the non-staggered σ_max, giving a less-absorbing PML for velocity.
    ///
    /// This matches k-Wave's behavior and corrects the ≈ 20% amplitude under-prediction
    /// that occurs when non-staggered sigma is applied to staggered velocity fields.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn apply_velocity_pml_directional(
        &mut self,
        mut field: ArrayViewMut3<f64>,
        grid: &Grid,
        _time_step: usize,
        axis: usize,
    ) -> KwaversResult<()> {
        let dt = self.estimate_dt_from_grid(grid);
        Zip::indexed(&mut field).par_for_each(|(i, j, k), val| {
            let sigma = match axis {
                0 => self.profiles.sigma_x_sgx[i],
                1 => self.profiles.sigma_y_sgy[j],
                _ => self.profiles.sigma_z_sgz[k],
            };
            if sigma > 0.0 {
                *val *= (-sigma * dt * 0.5).exp();
            }
        });
        Ok(())
    }

    fn apply_light(&mut self, _field: ArrayViewMut3<f64>, _grid: &Grid, _time_step: usize) {
        // CPML for light is not implemented yet
    }
}
