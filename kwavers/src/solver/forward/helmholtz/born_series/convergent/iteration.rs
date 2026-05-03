//! CBS solve loop, per-iteration step, and residual for `ConvergentBornSolver`.

use super::solver::ConvergentBornSolver;
use super::stats::ConvergentBornStats;
use crate::core::error::KwaversResult;
use crate::domain::medium::Medium;
use ndarray::{ArrayView3, ArrayViewMut3, Zip};
use num_complex::Complex64;

impl ConvergentBornSolver {
    /// Solve the Helmholtz equation using Convergent Born Series.
    pub fn solve<M: Medium>(
        &mut self,
        wavenumber: f64,
        medium: &M,
        incident_field: ArrayView3<Complex64>,
        mut result: ArrayViewMut3<Complex64>,
    ) -> KwaversResult<ConvergentBornStats> {
        self.incident_field.assign(&incident_field);
        self.current_field.assign(&incident_field);

        if self.green_fft.is_none() {
            self.precompute_green_function(wavenumber)?;
        }

        let mut stats = ConvergentBornStats::default();

        for iteration in 0..self.config.max_iterations {
            let residual = self.cbs_iteration(wavenumber, medium)?;
            stats.iterations = iteration + 1;
            stats.final_residual = residual;

            if residual < self.config.tolerance {
                stats.converged = true;
                break;
            }
        }

        result.assign(&self.current_field);
        Ok(stats)
    }

    /// Perform one CBS iteration: `ψ_{n+1} = ψ_n + G * (-k² V ψ_n)`.
    fn cbs_iteration<M: Medium>(&mut self, wavenumber: f64, medium: &M) -> KwaversResult<f64> {
        let k_squared = wavenumber * wavenumber;

        Zip::indexed(&mut self.workspace.heterogeneity_workspace).for_each(
            |(i, j, k), heterogeneity| {
                let current_val = self.current_field[[i, j, k]];
                let c_local = medium.sound_speed(i, j, k);
                let rho_local = medium.density(i, j, k);
                let c0 = 1500.0_f64;
                let rho0 = 1000.0_f64;
                let contrast = (rho_local * c_local * c_local) / (rho0 * c0 * c0);
                let v = k_squared * (1.0 - contrast);
                *heterogeneity = Complex64::new(v, 0.0) * current_val;
            },
        );

        self.apply_green_operator()?;

        let scale = Complex64::new(-k_squared, 0.0);
        self.workspace
            .green_workspace
            .par_mapv_inplace(|x| x * scale);

        Zip::from(&mut self.current_field)
            .and(&self.workspace.green_workspace)
            .for_each(|current, &update| {
                *current += update;
            });

        Ok(self.compute_residual())
    }

    /// Compute normalized L² residual over the scattering-potential workspace.
    fn compute_residual(&self) -> f64 {
        let mut residual = 0.0;
        Zip::from(&self.workspace.heterogeneity_workspace).for_each(|&scattering| {
            residual += scattering.norm_sqr();
        });
        (residual / (self.grid.nx * self.grid.ny * self.grid.nz) as f64).sqrt()
    }
}
