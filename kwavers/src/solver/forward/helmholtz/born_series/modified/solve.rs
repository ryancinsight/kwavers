use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use crate::core::error::KwaversResult;
use crate::domain::medium::Medium;
use ndarray::{Array3, ArrayView3, ArrayViewMut3};
use num_complex::Complex64;

use super::{ModifiedBornSolver, ModifiedBornStats};

impl ModifiedBornSolver {
    /// Solve.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn solve<M: Medium>(
        &mut self,
        wavenumber: f64,
        frequency: f64,
        medium: &M,
        incident_field: ArrayView3<Complex64>,
        mut result: ArrayViewMut3<Complex64>,
    ) -> KwaversResult<ModifiedBornStats> {
        self.precompute_viscoacoustic_properties(frequency, medium)?;

        self.workspace.clear();
        self.workspace.field_workspace.assign(&incident_field);

        let mut total_field = incident_field.to_owned();
        let mut stats = ModifiedBornStats::default();

        for order in 1..=self.config.max_order as usize {
            let scattering_field =
                self.compute_modified_born_term(wavenumber, frequency, medium, total_field.view())?;

            total_field += &scattering_field;

            let residual = self.compute_viscoacoustic_residual(
                wavenumber,
                frequency,
                medium,
                total_field.view(),
            );

            stats.orders_computed = order;
            stats.final_residual = residual;

            if residual < self.config.tolerance {
                stats.converged = true;
                break;
            }

            if order >= 10 {
                break;
            }
        }

        result.assign(&total_field);
        Ok(stats)
    }
    /// Compute modified born term.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn compute_modified_born_term<M: Medium>(
        &mut self,
        wavenumber: f64,
        frequency: f64,
        medium: &M,
        current_field: ArrayView3<Complex64>,
    ) -> KwaversResult<Array3<Complex64>> {
        let k_squared = wavenumber * wavenumber;
        let mut scattering_field =
            Array3::<Complex64>::zeros((self.grid.nx, self.grid.ny, self.grid.nz));

        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let v = self.compute_heterogeneity_potential(medium, i, j, k);
                    let absorption = self.absorption_field[[i, j, k]];

                    let source_term = -k_squared
                        * (Complex64::new(1.0, 0.0) - absorption)
                        * v
                        * current_field[[i, j, k]];

                    self.workspace.heterogeneity_workspace[[i, j, k]] = source_term;
                }
            }
        }

        self.apply_viscoacoustic_green(wavenumber, frequency)?;
        scattering_field.assign(&self.workspace.green_workspace);

        Ok(scattering_field)
    }

    pub(super) fn compute_heterogeneity_potential<M: Medium>(
        &self,
        medium: &M,
        i: usize,
        j: usize,
        k: usize,
    ) -> f64 {
        let c_local = medium.sound_speed(i, j, k);
        let rho_local = medium.density(i, j, k);

        let c0 = SOUND_SPEED_WATER_SIM;
        let rho0 = DENSITY_WATER_NOMINAL;

        1.0 - (rho_local * c_local * c_local) / (rho0 * c0 * c0)
    }

    pub(super) fn compute_viscoacoustic_residual<M: Medium>(
        &self,
        wavenumber: f64,
        _frequency: f64,
        medium: &M,
        field: ArrayView3<Complex64>,
    ) -> f64 {
        let mut residual_sum = 0.0;

        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let field_val = field[[i, j, k]];

                    let laplacian = self.compute_laplacian(field.view(), i, j, k);

                    let absorption = self.absorption_field[[i, j, k]];
                    let heterogeneity = self.compute_heterogeneity_potential(medium, i, j, k);

                    let k_complex = Complex64::new(wavenumber, absorption.im);
                    let helmholtz_term = k_complex
                        * k_complex
                        * (Complex64::new(1.0 + heterogeneity, 0.0) - absorption);

                    let residual = laplacian + helmholtz_term * field_val;
                    residual_sum += residual.norm_sqr();
                }
            }
        }

        (residual_sum / (self.grid.nx * self.grid.ny * self.grid.nz) as f64).sqrt()
    }
}
