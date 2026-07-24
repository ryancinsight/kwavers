//! Unified thermal diffusion solver.
//!
//! The parabolic thermal diffusion and Pennes bioheat branches share the same
//! centered finite-difference Laplacian. The Cattaneo-Vernotte branch is
//! hyperbolic, but still belongs to this thermal-wave solver family.
//!
//! ## Finite-difference theorem
//!
//! A centered second-derivative stencil with coefficients `c_m` satisfying
//! `Σ c_m = 0` and `Σ m² c_m = 2` differentiates any quadratic coordinate field
//! exactly. Therefore the discrete Laplacian of `x² + 2y² + 3z²` is `12` on all
//! points with complete stencil support. The regression tests use this
//! invariant because it is independent of timestep, material parameters, and
//! boundary treatment.

use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use leto::{Array3, ArrayView3};
use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};

use kwavers_physics::thermal::diffusion::{
    BioheatParameters, CattaneoVernotte, HyperbolicParameters, PennesBioheat,
    ThermalDiffusionConfig, ThermalDoseCalculator,
};

#[derive(Debug)]
pub struct ThermalDiffusionSolver {
    config: ThermalDiffusionConfig,
    temperature: Array3<f64>,
    temperature_prev: Option<Array3<f64>>,
    bioheat_solver: Option<PennesBioheat>,
    hyperbolic_solver: Option<CattaneoVernotte>,
    dose_calculator: Option<ThermalDoseCalculator>,
    pub(super) laplacian_workspace: Array3<f64>,
    current_time: f64,
}

impl ThermalDiffusionSolver {
    pub fn new(config: ThermalDiffusionConfig, grid: &Grid) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);
        let temperature = Array3::from_elem(shape, config.arterial_temperature.into_base());

        let bioheat_solver = if config.enable_bioheat {
            Some(PennesBioheat::new(BioheatParameters {
                perfusion_rate: config.perfusion_rate,
                blood_density: config.blood_density,
                blood_specific_heat: config.blood_specific_heat,
                arterial_temperature: config.arterial_temperature,
            }))
        } else {
            None
        };

        let hyperbolic_solver = if config.enable_hyperbolic {
            Some(CattaneoVernotte::new(
                HyperbolicParameters {
                    relaxation_time: config.relaxation_time,
                },
                grid,
            ))
        } else {
            None
        };

        let dose_calculator = if config.track_thermal_dose {
            Some(ThermalDoseCalculator::new(shape))
        } else {
            None
        };

        Self {
            config,
            temperature,
            temperature_prev: None,
            bioheat_solver,
            hyperbolic_solver,
            dose_calculator,
            laplacian_workspace: Array3::zeros(shape),
            current_time: 0.0,
        }
    }

    /// Compute `∇²T` into the pre-allocated Laplacian workspace.
    ///
    /// # Errors
    /// Returns [`crate::KwaversError::Validation`] if `spatial_order` is not 2 or 4.
    /// Invalid orders are rejected rather than downgraded because changing the
    /// stencil order changes the truncation-error and stability contract.
    fn calculate_laplacian(&mut self, grid: &Grid) -> KwaversResult<()> {
        match self.config.spatial_order {
            2 => self.calculate_laplacian_order::<2>(grid),
            4 => self.calculate_laplacian_order::<4>(grid),
            _ => {
                return Err(KwaversError::Validation(ValidationError::InvalidValue {
                    parameter: "spatial_order".to_owned(),
                    value: self.config.spatial_order as f64,
                    reason: "thermal diffusion supports only 2 or 4".to_owned(),
                }));
            }
        }

        Ok(())
    }

    /// Compute the finite-difference Laplacian through one monomorphized stencil body.
    ///
    /// # Contract
    /// `ORDER` selects the maximum centered stencil width at compile time.
    /// Each axis then independently selects the widest admissible centered
    /// second derivative:
    /// - `n >= 5` and `ORDER == 4`: fourth-order centered derivative.
    /// - `n >= 3`: second-order centered derivative.
    /// - `n == 1`: inactive embedded dimension, derivative contribution is zero.
    /// - `n == 2`: no centered second derivative exists, so no interior cells are updated.
    ///
    /// This keeps one authoritative algorithm while allowing the optimizer to
    /// eliminate the order branch per instantiation.
    #[inline]
    fn calculate_laplacian_order<const ORDER: usize>(&mut self, grid: &Grid) {
        let i_range = Self::axis_range::<ORDER>(grid.nx);
        let j_range = Self::axis_range::<ORDER>(grid.ny);
        let k_range = Self::axis_range::<ORDER>(grid.nz);

        let dx2_inv = 1.0 / (grid.dx * grid.dx);
        let dy2_inv = 1.0 / (grid.dy * grid.dy);
        let dz2_inv = 1.0 / (grid.dz * grid.dz);

        for i in i_range {
            for j in j_range.clone() {
                for k in k_range.clone() {
                    let d2_dx2 = Self::second_derivative_axis::<0, ORDER>(
                        &self.temperature,
                        i,
                        j,
                        k,
                        dx2_inv,
                        grid.nx,
                    );
                    let d2_dy2 = Self::second_derivative_axis::<1, ORDER>(
                        &self.temperature,
                        i,
                        j,
                        k,
                        dy2_inv,
                        grid.ny,
                    );
                    let d2_dz2 = Self::second_derivative_axis::<2, ORDER>(
                        &self.temperature,
                        i,
                        j,
                        k,
                        dz2_inv,
                        grid.nz,
                    );

                    self.laplacian_workspace[[i, j, k]] = d2_dx2 + d2_dy2 + d2_dz2;
                }
            }
        }
    }

    #[inline]
    fn axis_range<const ORDER: usize>(n: usize) -> std::ops::Range<usize> {
        if n == 1 {
            0..1
        } else if ORDER >= 4 && n >= 5 {
            2..n - 2
        } else if n >= 3 {
            1..n - 1
        } else {
            0..0
        }
    }

    #[inline]
    fn second_derivative_axis<const AXIS: usize, const ORDER: usize>(
        field: &Array3<f64>,
        i: usize,
        j: usize,
        k: usize,
        inv_h2: f64,
        n: usize,
    ) -> f64 {
        if ORDER >= 4 && n >= 5 {
            Self::second_derivative_fourth::<AXIS>(field, i, j, k, inv_h2)
        } else if n >= 3 {
            Self::second_derivative_second::<AXIS>(field, i, j, k, inv_h2)
        } else {
            0.0
        }
    }

    #[inline]
    fn second_derivative_second<const AXIS: usize>(
        field: &Array3<f64>,
        i: usize,
        j: usize,
        k: usize,
        inv_h2: f64,
    ) -> f64 {
        let center = field[[i, j, k]];
        let (hi, lo) = match AXIS {
            0 => (field[[i + 1, j, k]], field[[i - 1, j, k]]),
            1 => (field[[i, j + 1, k]], field[[i, j - 1, k]]),
            2 => (field[[i, j, k + 1]], field[[i, j, k - 1]]),
            _ => unreachable!("AXIS is a const-generic selector in 0..3"),
        };
        (2.0f64.mul_add(-center, hi) + lo) * inv_h2
    }

    #[inline]
    fn second_derivative_fourth<const AXIS: usize>(
        field: &Array3<f64>,
        i: usize,
        j: usize,
        k: usize,
        inv_h2: f64,
    ) -> f64 {
        const C0: f64 = -1.0 / 12.0;
        const C1: f64 = 4.0 / 3.0;
        const C2: f64 = -5.0 / 2.0;

        let center = field[[i, j, k]];
        let (p2, p1, m1, m2) = match AXIS {
            0 => (
                field[[i + 2, j, k]],
                field[[i + 1, j, k]],
                field[[i - 1, j, k]],
                field[[i - 2, j, k]],
            ),
            1 => (
                field[[i, j + 2, k]],
                field[[i, j + 1, k]],
                field[[i, j - 1, k]],
                field[[i, j - 2, k]],
            ),
            2 => (
                field[[i, j, k + 2]],
                field[[i, j, k + 1]],
                field[[i, j, k - 1]],
                field[[i, j, k - 2]],
            ),
            _ => unreachable!("AXIS is a const-generic selector in 0..3"),
        };

        C0.mul_add(
            p2,
            C1.mul_add(p1, C2.mul_add(center, C1.mul_add(m1, C0 * m2))),
        ) * inv_h2
    }

    /// Update.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn update(
        &mut self,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
        external_source: Option<ArrayView3<'_, f64>>,
    ) -> KwaversResult<()> {
        if self.temperature_prev.is_none() {
            self.temperature_prev = Some(Array3::zeros(self.temperature.shape()));
        }
        if let Some(ref mut prev) = self.temperature_prev {
            prev.assign(&self.temperature);
        }

        if self.config.enable_hyperbolic {
            if let Some(ref mut solver) = self.hyperbolic_solver {
                solver.update_temperature(&mut self.temperature, medium, grid, dt)?;
            }
        } else {
            self.calculate_laplacian(grid)?;

            if self.config.enable_bioheat {
                if let Some(ref solver) = self.bioheat_solver {
                    solver.update(
                        &mut self.temperature,
                        &self.laplacian_workspace,
                        external_source,
                        medium,
                        grid,
                        dt,
                    )?;
                }
            } else {
                self.update_standard_diffusion(external_source, medium, grid, dt)?;
            }
        }

        if let Some(ref mut calc) = self.dose_calculator {
            self.current_time += dt;
            calc.update_dose(&self.temperature, dt, self.current_time)?;
        }

        Ok(())
    }

    fn update_standard_diffusion(
        &mut self,
        external_source: Option<ArrayView3<'_, f64>>,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        let shape = self.temperature.shape();
        if let Some(source) = external_source.as_ref() {
            if source.shape() != shape {
                return Err(KwaversError::DimensionMismatch(format!(
                    "thermal diffusion source shape {:?} does not match temperature shape {:?}",
                    source.shape(),
                    shape
                )));
            }
        }

        let [_, ny, nz] = shape;
        let slab_len = ny * nz;
        let source_slice = external_source
            .as_ref()
            .and_then(|source| source.as_slice());

        let update_cell = |i: usize, j: usize, k: usize, temp: &mut f64, lap: f64, source: f64| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;

            let alpha = medium.thermal_diffusivity(x, y, z, grid);

            *temp += dt * alpha.mul_add(lap, source);
        };

        let source_is_contiguous = external_source.is_none() || source_slice.is_some();

        if let (Some(temperature), Some(laplacian), true) = (
            self.temperature.as_slice_mut(),
            self.laplacian_workspace.as_slice(),
            source_is_contiguous,
        ) {
            for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                temperature,
                slab_len,
                |i, slab| {
                    let base = i * slab_len;
                    for (offset, temp) in slab.iter_mut().enumerate() {
                        let j = offset / nz;
                        let k = offset % nz;
                        let source = source_slice.map_or(0.0, |source| source[base + offset]);
                        update_cell(i, j, k, temp, laplacian[base + offset], source);
                    }
                },
            );
        } else {
            let laplacian_workspace = &self.laplacian_workspace;
            self.temperature
                .indexed_iter_mut()
                .expect("invariant: contiguous owned temperature array")
                .for_each(|([i, j, k], temp)| {
                    let lap = laplacian_workspace[[i, j, k]];
                    let source = external_source.as_ref().map_or(0.0, |s| s[[i, j, k]]);
                    update_cell(i, j, k, temp, lap, source);
                });
        }

        Ok(())
    }

    #[must_use]
    pub fn temperature(&self) -> &Array3<f64> {
        &self.temperature
    }

    pub fn set_temperature(&mut self, temperature: Array3<f64>) {
        self.temperature = temperature;
    }

    #[must_use]
    pub fn thermal_dose(&self) -> Option<&Array3<f64>> {
        self.dose_calculator
            .as_ref()
            .map(ThermalDoseCalculator::get_dose)
    }

    #[must_use]
    pub fn max_thermal_dose(&self) -> Option<f64> {
        self.dose_calculator
            .as_ref()
            .map(ThermalDoseCalculator::max_dose)
    }

    #[must_use]
    pub fn necrosis_fraction(&self) -> Option<f64> {
        self.dose_calculator
            .as_ref()
            .map(ThermalDoseCalculator::necrosis_fraction)
    }

    pub fn reset(&mut self, grid: &Grid) {
        let shape = (grid.nx, grid.ny, grid.nz);
        self.temperature = Array3::from_elem(shape, self.config.arterial_temperature.into_base());
        self.temperature_prev = None;
        self.laplacian_workspace.fill(0.0);
        self.current_time = 0.0;

        if let Some(ref mut calc) = self.dose_calculator {
            calc.reset();
        }
    }
}

#[cfg(test)]
mod tests;
