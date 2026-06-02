//! Cattaneo-Vernotte Hyperbolic Heat Transfer
//!
//! Reference: Cattaneo, C. (1958). "A form of heat conduction equation which eliminates
//! the paradox of instantaneous propagation." Comptes Rendus, 247, 431-433.

use kwavers_core::error::KwaversResult;
use kwavers_domain::grid::Grid;
use kwavers_domain::medium::Medium;
use ndarray::{Array3, Zip};
use std::ops::Range;

/// Parameters for the Cattaneo-Vernotte hyperbolic heat update.
///
/// Only the thermal relaxation time `τ` enters the flux relaxation law
/// (`q + τ ∂q/∂t = −k∇T`); the thermal wave speed `c = √(α/τ)` is a *derived*
/// quantity (α = thermal diffusivity from the medium), not an independent input,
/// so it is intentionally not a field here — storing it would invite the
/// physically-inconsistent `(τ, c)` pairs the previous default carried.
///
/// Reference: Cattaneo (1958); Mitra et al. (1995), J. Heat Transfer 117, 568
/// (measured τ ≈ 16 s in processed tissue — the basis for the 20 s default).
#[derive(Debug, Clone)]
pub struct HyperbolicParameters {
    /// Thermal relaxation time `τ` [s].
    pub relaxation_time: f64,
}

impl Default for HyperbolicParameters {
    fn default() -> Self {
        Self {
            relaxation_time: 20.0,
        }
    }
}

#[derive(Debug)]
pub struct CattaneoVernotte {
    params: HyperbolicParameters,
    heat_flux_x: Array3<f64>,
    heat_flux_y: Array3<f64>,
    heat_flux_z: Array3<f64>,
    divergence: Array3<f64>,
}

impl CattaneoVernotte {
    pub fn new(params: HyperbolicParameters, grid: &Grid) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);
        Self {
            params,
            heat_flux_x: Array3::zeros(shape),
            heat_flux_y: Array3::zeros(shape),
            heat_flux_z: Array3::zeros(shape),
            divergence: Array3::zeros(shape),
        }
    }

    /// Update heat flux using the Cattaneo-Vernotte relaxation law.
    ///
    /// # Contract
    /// Each Cartesian component is independent under the centered-difference
    /// flux update. The `AXIS` const parameter used by the component helper
    /// selects the compile-time coordinate, so the compiler emits one inlined
    /// specialization per component without duplicated algorithm bodies.
    ///
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    pub fn update_heat_flux(
        &mut self,
        temperature: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        let tau = self.params.relaxation_time;

        Self::update_flux_axis::<0>(&mut self.heat_flux_x, temperature, medium, grid, dt, tau);
        Self::update_flux_axis::<1>(&mut self.heat_flux_y, temperature, medium, grid, dt, tau);
        Self::update_flux_axis::<2>(&mut self.heat_flux_z, temperature, medium, grid, dt, tau);

        Ok(())
    }

    /// Compute heat-flux divergence into a newly allocated array.
    ///
    /// `update_temperature` uses the internal workspace instead of this owned
    /// result to avoid one full-volume allocation per step.
    #[must_use]
    pub fn heat_flux_divergence(&self, grid: &Grid) -> Array3<f64> {
        let mut div = Array3::zeros((grid.nx, grid.ny, grid.nz));
        Self::fill_heat_flux_divergence(
            &self.heat_flux_x,
            &self.heat_flux_y,
            &self.heat_flux_z,
            grid,
            &mut div,
        );
        div
    }

    /// Update temperature.
    ///
    /// # Contract
    /// The divergence field is a reusable workspace owned by the solver. Its
    /// values depend only on the current heat-flux components and grid spacing,
    /// so reuse is observationally equivalent to allocating a fresh divergence
    /// array for each explicit Euler step.
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    pub fn update_temperature(
        &mut self,
        temperature: &mut Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        self.update_heat_flux(temperature, medium, grid, dt)?;
        Self::fill_heat_flux_divergence(
            &self.heat_flux_x,
            &self.heat_flux_y,
            &self.heat_flux_z,
            grid,
            &mut self.divergence,
        );

        Zip::indexed(temperature)
            .and(&self.divergence)
            .par_for_each(|(i, j, k), t, &div| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let rho = kwavers_domain::medium::density_at(medium, x, y, z, grid);
                let cp = medium.specific_heat(x, y, z, grid);

                *t -= dt * div / (rho * cp);
            });

        Ok(())
    }

    #[inline]
    fn update_flux_axis<const AXIS: usize>(
        flux: &mut Array3<f64>,
        temperature: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
        tau: f64,
    ) {
        let relax = dt / tau;
        let denominator = 1.0 + relax;

        Zip::indexed(flux).par_for_each(|(i, j, k), q| {
            if !Self::has_centered_neighbor::<AXIS>(i, j, k, grid) {
                *q = 0.0;
                return;
            }

            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;
            let k_thermal = medium.thermal_conductivity(x, y, z, grid);
            let grad_t = Self::centered_gradient::<AXIS>(temperature, i, j, k, grid);
            let q_previous = *q;

            // Backward Euler: τ·∂q/∂t + q = −k·∇T  ⟹  q^{n+1} = (q^n − relax·k·∇T) / (1 + relax)
            *q = (q_previous - relax * k_thermal * grad_t) / denominator;
        });
    }

    fn fill_heat_flux_divergence(
        flux_x: &Array3<f64>,
        flux_y: &Array3<f64>,
        flux_z: &Array3<f64>,
        grid: &Grid,
        divergence: &mut Array3<f64>,
    ) {
        let x_range = Self::centered_axis_range(grid.nx);
        let y_range = Self::centered_axis_range(grid.ny);
        let z_range = Self::centered_axis_range(grid.nz);

        Zip::indexed(divergence).par_for_each(|(i, j, k), div| {
            if !x_range.contains(&i) || !y_range.contains(&j) || !z_range.contains(&k) {
                *div = 0.0;
                return;
            }

            let div_x = Self::divergence_component::<0>(flux_x, i, j, k, grid);
            let div_y = Self::divergence_component::<1>(flux_y, i, j, k, grid);
            let div_z = Self::divergence_component::<2>(flux_z, i, j, k, grid);

            *div = div_x + div_y + div_z;
        });
    }

    #[inline]
    fn centered_axis_range(n: usize) -> Range<usize> {
        match n {
            0 | 2 => 0..0,
            1 => 0..1,
            _ => 1..(n - 1),
        }
    }

    #[inline]
    fn has_centered_neighbor<const AXIS: usize>(i: usize, j: usize, k: usize, grid: &Grid) -> bool {
        match AXIS {
            0 => Self::has_axis_neighbor(i, grid.nx),
            1 => Self::has_axis_neighbor(j, grid.ny),
            2 => Self::has_axis_neighbor(k, grid.nz),
            _ => unreachable!("axis must be 0, 1, or 2"),
        }
    }

    #[inline]
    const fn has_axis_neighbor(index: usize, n: usize) -> bool {
        index > 0 && index + 1 < n
    }

    #[inline]
    fn centered_gradient<const AXIS: usize>(
        temperature: &Array3<f64>,
        i: usize,
        j: usize,
        k: usize,
        grid: &Grid,
    ) -> f64 {
        match AXIS {
            0 => (temperature[[i + 1, j, k]] - temperature[[i - 1, j, k]]) / (2.0 * grid.dx),
            1 => (temperature[[i, j + 1, k]] - temperature[[i, j - 1, k]]) / (2.0 * grid.dy),
            2 => (temperature[[i, j, k + 1]] - temperature[[i, j, k - 1]]) / (2.0 * grid.dz),
            _ => unreachable!("axis must be 0, 1, or 2"),
        }
    }

    #[inline]
    fn divergence_component<const AXIS: usize>(
        flux: &Array3<f64>,
        i: usize,
        j: usize,
        k: usize,
        grid: &Grid,
    ) -> f64 {
        if !Self::has_centered_neighbor::<AXIS>(i, j, k, grid) {
            return 0.0;
        }

        match AXIS {
            0 => (flux[[i + 1, j, k]] - flux[[i - 1, j, k]]) / (2.0 * grid.dx),
            1 => (flux[[i, j + 1, k]] - flux[[i, j - 1, k]]) / (2.0 * grid.dy),
            2 => (flux[[i, j, k + 1]] - flux[[i, j, k - 1]]) / (2.0 * grid.dz),
            _ => unreachable!("axis must be 0, 1, or 2"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use kwavers_domain::medium::homogeneous::HomogeneousMedium;

    fn homogeneous_unit_medium(grid: &Grid) -> HomogeneousMedium {
        let mut medium = HomogeneousMedium::new(1.0, SOUND_SPEED_WATER_SIM, 0.0, 0.0, grid);
        medium
            .set_thermal_properties(1.0, 1.0)
            .expect("unit thermal properties are valid");
        medium
    }

    #[test]
    fn update_temperature_reuses_divergence_workspace_for_one_dimensional_grid() {
        let grid = Grid::new(5, 1, 1, 1.0, 1.0, 1.0).expect("valid one-dimensional grid");
        let medium = homogeneous_unit_medium(&grid);
        let mut solver = CattaneoVernotte::new(
            HyperbolicParameters {
                relaxation_time: 1.0,
            },
            &grid,
        );
        let workspace_ptr = solver.divergence.as_ptr();
        let mut temperature =
            Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |(i, _, _)| (i * i) as f64);
        let center_before = temperature[[2, 0, 0]];
        let dt = 0.1;
        let relax = dt / solver.params.relaxation_time;
        let expected_center_after_one_step = center_before + dt * (2.0 * relax / (1.0 + relax));

        solver
            .update_temperature(&mut temperature, &medium, &grid, dt)
            .expect("hyperbolic update succeeds");
        let center_after_one_step = temperature[[2, 0, 0]];
        let tolerance = 8.0 * f64::EPSILON * expected_center_after_one_step.abs().max(1.0);

        assert!((center_after_one_step - expected_center_after_one_step).abs() <= tolerance);
        assert_eq!(solver.divergence.as_ptr(), workspace_ptr);

        solver
            .update_temperature(&mut temperature, &medium, &grid, dt)
            .expect("hyperbolic update succeeds");

        assert_eq!(solver.divergence.as_ptr(), workspace_ptr);
    }

    #[test]
    fn owned_heat_flux_divergence_matches_workspace_fill() {
        let grid = Grid::new(5, 5, 5, 1.0, 1.0, 1.0).expect("valid grid");
        let mut solver = CattaneoVernotte::new(HyperbolicParameters::default(), &grid);

        Zip::indexed(&mut solver.heat_flux_x).par_for_each(|(i, _, _), q| *q = i as f64);
        Zip::indexed(&mut solver.heat_flux_y).par_for_each(|(_, j, _), q| *q = (2 * j) as f64);
        Zip::indexed(&mut solver.heat_flux_z).par_for_each(|(_, _, k), q| *q = (3 * k) as f64);

        let owned = solver.heat_flux_divergence(&grid);
        CattaneoVernotte::fill_heat_flux_divergence(
            &solver.heat_flux_x,
            &solver.heat_flux_y,
            &solver.heat_flux_z,
            &grid,
            &mut solver.divergence,
        );

        assert_eq!(owned, solver.divergence);
        assert_eq!(owned[[0, 2, 2]], 0.0);
        assert_eq!(owned[[2, 2, 2]], 6.0);
    }
}
