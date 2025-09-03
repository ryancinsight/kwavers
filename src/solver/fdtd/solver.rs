//! Core FDTD solver implementation
//!
//! This module contains the main `FdtdSolver` struct and its implementation
//! for acoustic wave propagation using the finite-difference time-domain method.

use crate::boundary::cpml::CPMLBoundary;
use crate::error::{KwaversError, KwaversResult, ValidationError};
use crate::grid::Grid;
use crate::physics::mechanics::acoustic_wave::SpatialOrder;
use log::info;
use ndarray::{s, Array3, ArrayView3, Zip};

use super::config::FdtdConfig;
use super::finite_difference::FiniteDifference;
use super::metrics::FdtdMetrics;
use super::staggered_grid::StaggeredGrid;

/// FDTD solver for acoustic wave propagation
#[derive(Clone, Debug)]
pub struct FdtdSolver {
    /// Configuration
    pub(crate) config: FdtdConfig,
    /// Grid reference
    pub(crate) grid: Grid,
    /// Staggered grid positions
    pub(crate) staggered: StaggeredGrid,
    /// Finite difference operator
    pub(crate) fd_operator: FiniteDifference,
    /// Performance metrics
    metrics: FdtdMetrics,
    /// C-PML boundary (if enabled)
    pub(crate) cpml_boundary: Option<CPMLBoundary>,
    /// Spatial order enum (validated at construction)
    spatial_order: SpatialOrder,
}

impl FdtdSolver {
    /// Create a new FDTD solver
    pub fn new(config: FdtdConfig, grid: &Grid) -> KwaversResult<Self> {
        info!("Initializing FDTD solver with config: {:?}", config);

        // Validate spatial order by converting to enum
        let spatial_order = SpatialOrder::from_usize(config.spatial_order)
            .map_err(|_| KwaversError::Validation(ValidationError::FieldValidation {
                field: "spatial_order".to_string(),
                value: config.spatial_order.to_string(),
                constraint: "must be a supported order (e.g., 2, 4, 6)".to_string(),
            }))?;

        // Create finite difference operator
        let fd_operator = FiniteDifference::new(config.spatial_order)?;

        Ok(Self {
            config,
            grid: grid.clone(),
            staggered: StaggeredGrid::default(),
            fd_operator,
            metrics: FdtdMetrics::new(),
            cpml_boundary: None,
            spatial_order,
        })
    }

    /// Enable C-PML boundary conditions
    pub fn enable_cpml(
        &mut self,
        config: crate::boundary::cpml::CPMLConfig,
        _dt: f64,
        max_sound_speed: f64,
    ) -> KwaversResult<()> {
        info!("Enabling C-PML boundary conditions");
        self.cpml_boundary = Some(CPMLBoundary::new(
            config,
            &self.grid,
            max_sound_speed,
        )?);
        Ok(())
    }

    /// Update pressure field using velocity divergence
    pub fn update_pressure(
        &mut self,
        pressure: &mut Array3<f64>,
        vx: &Array3<f64>,
        vy: &Array3<f64>,
        vz: &Array3<f64>,
        density: ArrayView3<f64>,
        sound_speed: ArrayView3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        // Compute divergence of velocity
        let divergence = self.fd_operator.compute_divergence(
            &vx.view(),
            &vy.view(),
            &vz.view(),
            self.grid.dx,
            self.grid.dy,
            self.grid.dz,
        )?;

        // Update pressure: p^{n+1} = p^n - dt * rho * c^2 * div(v)
        Zip::from(pressure)
            .and(&divergence)
            .and(density)
            .and(sound_speed)
            .for_each(|p, &div, &rho, &c| {
                *p -= dt * rho * c * c * div;
            });

        // Apply C-PML if enabled
        // Note: C-PML boundary conditions are applied to the gradient terms
        // in the velocity update, not directly to pressure

        Ok(())
    }

    /// Update velocity field using pressure gradient
    pub fn update_velocity(
        &mut self,
        vx: &mut Array3<f64>,
        vy: &mut Array3<f64>,
        vz: &mut Array3<f64>,
        pressure: &Array3<f64>,
        density: ArrayView3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        // Compute pressure gradient
        let (mut grad_x, mut grad_y, mut grad_z) = self.fd_operator.compute_gradient(
            &pressure.view(),
            self.grid.dx,
            self.grid.dy,
            self.grid.dz,
        )?;

        // Apply C-PML if enabled
        if let Some(ref mut cpml) = self.cpml_boundary {
            // Update C-PML memory variables for each component
            cpml.update_acoustic_memory(&grad_x, 0);
            cpml.update_acoustic_memory(&grad_y, 1);
            cpml.update_acoustic_memory(&grad_z, 2);

            // Apply C-PML to gradients
            cpml.apply_cpml_gradient(&mut grad_x, 0);
            cpml.apply_cpml_gradient(&mut grad_y, 1);
            cpml.apply_cpml_gradient(&mut grad_z, 2);
        }

        // Update velocity: v^{n+1/2} = v^{n-1/2} - dt/rho * grad(p)
        Zip::from(vx)
            .and(&grad_x)
            .and(density)
            .for_each(|v, &grad, &rho| {
                *v -= dt / rho * grad;
            });

        Zip::from(vy)
            .and(&grad_y)
            .and(density)
            .for_each(|v, &grad, &rho| {
                *v -= dt / rho * grad;
            });

        Zip::from(vz)
            .and(&grad_z)
            .and(density)
            .for_each(|v, &grad, &rho| {
                *v -= dt / rho * grad;
            });

        Ok(())
    }

    /// Interpolate field to staggered grid positions
    pub fn interpolate_to_staggered(
        &self,
        field: &ndarray::ArrayView3<f64>,
        axis: usize,
        offset: f64,
    ) -> KwaversResult<Array3<f64>> {
        if offset == 0.0 {
            return Ok(field.to_owned());
        }
        if offset != 0.5 {
            return Err(KwaversError::NotImplemented("Only 0.5 offset supported".to_string()));
        }

        let mut interpolated = Array3::zeros(field.raw_dim());
        let (nx, ny, nz) = field.dim();

        match axis {
            0 => {
                if nx < 2 { return Ok(interpolated); }
                let mut interp_slice = interpolated.slice_mut(s![..nx - 1, .., ..]);
                let s1 = field.slice(s![..nx - 1, .., ..]);
                let s2 = field.slice(s![1.., .., ..]);
                interp_slice.assign(&((&s1 + &s2) * 0.5));
                interpolated.slice_mut(s![nx - 1, .., ..]).assign(&field.slice(s![nx - 1, .., ..]));
            }
            1 => {
                if ny < 2 { return Ok(interpolated); }
                let mut interp_slice = interpolated.slice_mut(s![.., ..ny - 1, ..]);
                let s1 = field.slice(s![.., ..ny - 1, ..]);
                let s2 = field.slice(s![.., 1.., ..]);
                interp_slice.assign(&((&s1 + &s2) * 0.5));
                interpolated.slice_mut(s![.., ny - 1, ..]).assign(&field.slice(s![.., ny - 1, ..]));
            }
            2 => {
                if nz < 2 { return Ok(interpolated); }
                let mut interp_slice = interpolated.slice_mut(s![.., .., ..nz - 1]);
                let s1 = field.slice(s![.., .., ..nz - 1]);
                let s2 = field.slice(s![.., .., 1..]);
                interp_slice.assign(&((&s1 + &s2) * 0.5));
                interpolated.slice_mut(s![.., .., nz - 1]).assign(&field.slice(s![.., .., nz - 1]));
            }
            _ => {
                return Err(KwaversError::Validation(ValidationError::FieldValidation {
                    field: "axis".to_string(),
                    value: axis.to_string(),
                    constraint: "must be 0, 1, or 2".to_string(),
                }));
            }
        }
        Ok(interpolated)
    }

    /// Calculate maximum stable time step based on CFL condition
    pub fn max_stable_dt(&self, max_sound_speed: f64) -> f64 {
        let min_dx = self.grid.dx.min(self.grid.dy).min(self.grid.dz);
        let cfl_limit = self.spatial_order.cfl_limit();
        self.config.cfl_factor * cfl_limit * min_dx / max_sound_speed
    }

    /// Check if given timestep satisfies CFL condition
    pub fn check_cfl_stability(&self, dt: f64, max_sound_speed: f64) -> bool {
        let max_dt = self.max_stable_dt(max_sound_speed);
        dt <= max_dt
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> &FdtdMetrics {
        &self.metrics
    }

    /// Merge metrics from another solver instance
    pub fn merge_metrics(&mut self, other_metrics: &FdtdMetrics) {
        self.metrics.merge(other_metrics);
    }
}
