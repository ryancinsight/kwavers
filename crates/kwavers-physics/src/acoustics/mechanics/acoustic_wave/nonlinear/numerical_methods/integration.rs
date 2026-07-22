use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use leto::Array3;
use log::debug;
use std::time::Instant;

use super::super::wave_model::NonlinearWave;

impl NonlinearWave {
    /// Updates the wave field using the nonlinear acoustic wave equation.
    ///
    /// This method implements a pseudo-spectral time-domain (PSTD) solver for the
    /// nonlinear acoustic wave equation. It uses FFT for spatial derivatives and
    /// includes nonlinear terms for accurate modeling of high-intensity acoustics.
    ///
    /// # Arguments
    ///
    /// * `pressure` - Current pressure field \[Pa\]
    /// * `source` - Source term array [Pa/s²]
    /// * `medium` - Medium properties
    /// * `grid` - Computational grid
    /// * `time_step` - Current time step index
    ///
    /// # Returns
    ///
    /// Updated pressure field (internal implementation)
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn update_wave_inner(
        &mut self,
        pressure: &Array3<f64>,
        source: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
        time_step: usize,
    ) -> KwaversResult<Array3<f64>> {
        let start_total = Instant::now();
        self.call_count += 1;

        // Validate inputs
        if pressure.shape() != [grid.nx, grid.ny, grid.nz] {
            return Err(kwavers_core::error::KwaversError::InvalidInput(format!(
                "Pressure array shape [{}, {}, {}] doesn't match grid dimensions [{}, {}, {}]",
                pressure.shape()[0],
                pressure.shape()[1],
                pressure.shape()[2],
                grid.nx,
                grid.ny,
                grid.nz
            )));
        }

        // Compute nonlinear term
        let start_nonlinear = Instant::now();
        let nonlinear_term = self.compute_nonlinear_term(pressure, medium, grid)?;
        self.nonlinear_time += start_nonlinear.elapsed().as_secs_f64();

        // Apply k-space correction
        let start_fft = Instant::now();
        let linear_term = self.apply_k_space_correction(pressure, medium, grid)?;
        self.fft_time += start_fft.elapsed().as_secs_f64();

        let [nx, ny, nz] = source.shape();
        let mut updated_pressure = Array3::from_elem([nx, ny, nz], 0.0);

        // Combine terms
        let start_combination = Instant::now();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let nl = nonlinear_term[[i, j, k]] * self.nonlinearity_scaling;
                    let src = source[[i, j, k]] * self.dt.powi(2);
                    updated_pressure[[i, j, k]] = linear_term[[i, j, k]] + nl + src;
                }
            }
        }

        // Apply stability constraints if needed
        if self.clamp_gradients {
            self.apply_stability_constraints(&mut updated_pressure);
        }

        self.combination_time += start_combination.elapsed().as_secs_f64();

        debug!(
            "Step {}: max pressure = {:.2e} Pa, update time = {:.3} ms",
            time_step,
            updated_pressure
                .iter()
                .fold(0.0_f64, |max, &val| max.max(val.abs())),
            start_total.elapsed().as_secs_f64() * 1000.0
        );

        Ok(updated_pressure)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::wave_model::NonlinearWave;
    use kwavers_grid::Grid;
    use kwavers_medium::HomogeneousMedium;
    use leto::Array3;

    /// `update_wave_inner` must return Err when the pressure array shape does not
    /// match the grid dimensions.
    ///
    /// Precondition check: shape mismatch is detected before any FFT allocation.
    #[test]
    fn update_wave_inner_rejects_mismatched_pressure_shape() {
        let grid = Grid::new(4, 4, 4, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let mut w = NonlinearWave::new(&grid, 1e-7);

        // 3×3×3 ≠ 4×4×4
        let pressure = Array3::<f64>::zeros((3, 3, 3));
        let source = Array3::<f64>::zeros((4, 4, 4));

        let result = w.update_wave_inner(&pressure, &source, &medium, &grid, 0);
        assert!(result.is_err(), "shape mismatch must return Err");
    }

    /// `update_wave_inner` increments `call_count` on each invocation.
    #[test]
    fn update_wave_inner_increments_call_count() {
        let grid = Grid::new(4, 4, 4, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let mut w = NonlinearWave::new(&grid, 1e-7);
        w.precompute_k_squared(&grid);

        let pressure = Array3::<f64>::zeros((4, 4, 4));
        let source = Array3::<f64>::zeros((4, 4, 4));

        assert_eq!(w.call_count, 0);
        w.update_wave_inner(&pressure, &source, &medium, &grid, 0)
            .unwrap();
        assert_eq!(w.call_count, 1, "call_count must be 1 after one call");
        w.update_wave_inner(&pressure, &source, &medium, &grid, 1)
            .unwrap();
        assert_eq!(w.call_count, 2, "call_count must be 2 after two calls");
    }

    /// Zero source + zero pressure → output shape matches the grid dimensions.
    #[test]
    fn update_wave_inner_output_shape_matches_grid() {
        let grid = Grid::new(4, 4, 4, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let mut w = NonlinearWave::new(&grid, 1e-7);
        w.precompute_k_squared(&grid);

        let pressure = Array3::<f64>::zeros((4, 4, 4));
        let source = Array3::<f64>::zeros((4, 4, 4));

        let result = w
            .update_wave_inner(&pressure, &source, &medium, &grid, 0)
            .unwrap();
        assert_eq!(
            result.shape(),
            [4, 4, 4],
            "output shape must match grid (4,4,4), got {:?}",
            result.shape()
        );
    }
}