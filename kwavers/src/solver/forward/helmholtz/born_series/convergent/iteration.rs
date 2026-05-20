//! CBS solve loop, per-iteration step, and residual for `ConvergentBornSolver`.

use super::solver::ConvergentBornSolver;
use super::stats::ConvergentBornStats;
use crate::core::error::KwaversResult;
use crate::domain::medium::Medium;
use ndarray::{ArrayView3, ArrayViewMut3, Zip};
use num_complex::Complex64;
use rayon::prelude::*;

impl ConvergentBornSolver {
    /// Solve the Helmholtz equation using Convergent Born Series.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if `contrast shape`.
    ///
    fn cbs_iteration<M: Medium>(&mut self, wavenumber: f64, medium: &M) -> KwaversResult<f64> {
        let k_squared = wavenumber * wavenumber;
        let (nx, ny, nz) = self.workspace.heterogeneity_workspace.dim();
        let c0 = 1500.0_f64;
        let rho0 = 1000.0_f64;

        // Phase 1: sequential — medium properties are not guaranteed Sync.
        let contrasts: Vec<f64> = (0..nx)
            .flat_map(|i| {
                (0..ny).flat_map(move |j| {
                    (0..nz).map(move |k| {
                        let c = medium.sound_speed(i, j, k);
                        let rho = medium.density(i, j, k);
                        (rho * c * c) / (rho0 * c0 * c0)
                    })
                })
            })
            .collect();
        let contrasts_arr =
            ndarray::Array3::from_shape_vec((nx, ny, nz), contrasts).expect("contrast shape");

        // Phase 2: parallel — pure arithmetic on pre-collected medium values.
        let current_field = &self.current_field;
        Zip::from(&mut self.workspace.heterogeneity_workspace)
            .and(&contrasts_arr)
            .and(current_field)
            .par_for_each(|heterogeneity, &contrast, &current_val| {
                let v = 1.0 - contrast;
                *heterogeneity = Complex64::new(v, 0.0) * current_val;
            });

        self.apply_green_operator()?;

        let scale = Complex64::new(-k_squared, 0.0);
        self.workspace
            .green_workspace
            .par_mapv_inplace(|x| x * scale);

        Zip::from(&mut self.current_field)
            .and(&self.workspace.green_workspace)
            .par_for_each(|current, &update| {
                *current += update;
            });

        Ok(self.compute_residual())
    }

    /// Compute normalized L² residual over the scattering-potential workspace.
    /// # Panics
    /// - Panics if `heterogeneity_workspace contiguous`.
    ///
    fn compute_residual(&self) -> f64 {
        let residual: f64 = self
            .workspace
            .heterogeneity_workspace
            .as_slice()
            .expect("heterogeneity_workspace contiguous")
            .par_iter()
            .map(|s| s.norm_sqr())
            .sum();
        (residual / (self.grid.nx * self.grid.ny * self.grid.nz) as f64).sqrt()
    }
}
