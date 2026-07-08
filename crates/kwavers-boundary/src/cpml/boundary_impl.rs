//! `Boundary` trait impl for `CPMLBoundary`

use super::CPMLBoundary;
use crate::{Boundary, PmlExpFactors};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use ndarray::{Array3, ArrayViewMut3};

impl Boundary for CPMLBoundary {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn apply_acoustic(
        &mut self,
        field: ArrayViewMut3<f64>,
        grid: &Grid,
        _time_step: usize,
    ) -> KwaversResult<()> {
        // K-Wave applies PML as exp(-sigma * dt/2) to velocity AND density fields
        // (split-field PML, applied twice per step = net exp(-sigma*dt)).
        let dt = self.estimate_dt_from_grid(grid);

        let sigma_x = &self.profiles.sigma_x;
        let sigma_y = &self.profiles.sigma_y;
        let sigma_z = &self.profiles.sigma_z;
        crate::parallel::for_each_indexed_mut(field, |(i, j, k), val| {
            let s_x = sigma_x[i];
            let s_y = sigma_y[j];
            let s_z = sigma_z[k];
            let sigma_total = s_x + s_y + s_z;

            if sigma_total > 0.0 {
                *val *= (-sigma_total * dt * 0.5).exp();
            }
        });

        Ok(())
    }

    fn apply_acoustic_freq(
        &mut self,
        field: &mut Array3<kwavers_math::fft::Complex64>,
        grid: &Grid,
        _time_step: usize,
    ) -> KwaversResult<()> {
        let dt = self.estimate_dt_from_grid(grid);

        let sigma_x = &self.profiles.sigma_x;
        let sigma_y = &self.profiles.sigma_y;
        let sigma_z = &self.profiles.sigma_z;
        crate::parallel::for_each_indexed_mut(field.view_mut(), |(i, j, k), val| {
            let s_x = sigma_x[i];
            let s_y = sigma_y[j];
            let s_z = sigma_z[k];
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
        field: ArrayViewMut3<f64>,
        grid: &Grid,
        _time_step: usize,
        axis: usize,
    ) -> KwaversResult<()> {
        let dt = self.estimate_dt_from_grid(grid);
        let sigma_x = &self.profiles.sigma_x;
        let sigma_y = &self.profiles.sigma_y;
        let sigma_z = &self.profiles.sigma_z;
        crate::parallel::for_each_indexed_mut(field, |(i, j, k), val| {
            let sigma = match axis {
                0 => sigma_x[i],
                1 => sigma_y[j],
                _ => sigma_z[k],
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
        field: ArrayViewMut3<f64>,
        grid: &Grid,
        _time_step: usize,
        axis: usize,
    ) -> KwaversResult<()> {
        let dt = self.estimate_dt_from_grid(grid);
        let sigma_x = &self.profiles.sigma_x_sgx;
        let sigma_y = &self.profiles.sigma_y_sgy;
        let sigma_z = &self.profiles.sigma_z_sgz;
        crate::parallel::for_each_indexed_mut(field, |(i, j, k), val| {
            let sigma = match axis {
                0 => sigma_x[i],
                1 => sigma_y[j],
                _ => sigma_z[k],
            };
            if sigma > 0.0 {
                *val *= (-sigma * dt * 0.5).exp();
            }
        });
        Ok(())
    }

    /// Return precomputed split-field PML factors for fused velocity and density updates.
    ///
    /// Each factor array is a clone of the profile data computed at construction.
    /// The solver stores these on first call and never re-allocates per step.
    fn pml_exp_factors_owned(&self) -> Option<PmlExpFactors> {
        Some(PmlExpFactors {
            vel_x: self.profiles.pml_vel_x.clone(),
            vel_y: self.profiles.pml_vel_y.clone(),
            vel_z: self.profiles.pml_vel_z.clone(),
            den_x: self.profiles.pml_den_x.clone(),
            den_y: self.profiles.pml_den_y.clone(),
            den_z: self.profiles.pml_den_z.clone(),
        })
    }

    fn apply_light(&mut self, _field: ArrayViewMut3<f64>, _grid: &Grid, _time_step: usize) {
        // CPML for light is not implemented yet
    }
}
