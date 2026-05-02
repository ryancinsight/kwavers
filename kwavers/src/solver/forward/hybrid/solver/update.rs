use super::HybridSolver;
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::boundary::Boundary;
use crate::domain::medium::Medium;
use crate::domain::source::{Source, SourceField};
use crate::solver::forward::hybrid::config::DecompositionStrategy;
use crate::solver::forward::hybrid::domain_decomposition::{DomainRegion, DomainType};
use crate::solver::forward::hybrid::metrics::{HybridMetrics, ValidationResults};
use log::debug;
use ndarray::{s, Array4, Zip};
use std::time::Instant;

impl HybridSolver {
    /// Update fields for one time step
    pub fn update(
        &mut self,
        fields: &mut Array4<f64>,
        medium: &dyn Medium,
        source: &dyn Source,
        _boundary: &mut dyn Boundary,
        _dt: f64,
        t: f64,
    ) -> KwaversResult<()> {
        let update_start = Instant::now();

        if self.config.decomposition_strategy == DecompositionStrategy::Dynamic {
            self.update_decomposition(fields, medium)?;
        }

        use crate::domain::field::mapping::UnifiedFieldType;
        let p_idx = UnifiedFieldType::Pressure.index();
        let vx_idx = UnifiedFieldType::VelocityX.index();
        let vy_idx = UnifiedFieldType::VelocityY.index();
        let vz_idx = UnifiedFieldType::VelocityZ.index();

        self.pstd_solver
            .fields
            .p
            .assign(&fields.index_axis(ndarray::Axis(0), p_idx));
        self.pstd_solver
            .fields
            .ux
            .assign(&fields.index_axis(ndarray::Axis(0), vx_idx));
        self.pstd_solver
            .fields
            .uy
            .assign(&fields.index_axis(ndarray::Axis(0), vy_idx));
        self.pstd_solver
            .fields
            .uz
            .assign(&fields.index_axis(ndarray::Axis(0), vz_idx));

        self.fdtd_solver
            .fields
            .p
            .assign(&fields.index_axis(ndarray::Axis(0), p_idx));
        self.fdtd_solver
            .fields
            .ux
            .assign(&fields.index_axis(ndarray::Axis(0), vx_idx));
        self.fdtd_solver
            .fields
            .uy
            .assign(&fields.index_axis(ndarray::Axis(0), vy_idx));
        self.fdtd_solver
            .fields
            .uz
            .assign(&fields.index_axis(ndarray::Axis(0), vz_idx));

        let amp = source.amplitude(t);
        if amp.abs() > 1e-12 && source.source_type() == SourceField::Pressure {
            let mask = source.create_mask(&self.grid);
            Zip::from(&mut self.fdtd_solver.fields.p)
                .and(&mask)
                .for_each(|p, &m| *p += m * amp);
        }

        self.pstd_solver.step_forward()?;
        self.fdtd_solver.step_forward()?;

        let regions = self.regions.clone();
        for region in &regions {
            match region.domain_type {
                DomainType::PSTD => {
                    let mut p_view = fields.index_axis_mut(ndarray::Axis(0), p_idx);
                    let slice = s![
                        region.start.0..region.end.0,
                        region.start.1..region.end.1,
                        region.start.2..region.end.2
                    ];
                    p_view
                        .slice_mut(slice)
                        .assign(&self.pstd_solver.fields.p.slice(slice));

                    let mut vx_view = fields.index_axis_mut(ndarray::Axis(0), vx_idx);
                    vx_view
                        .slice_mut(slice)
                        .assign(&self.pstd_solver.fields.ux.slice(slice));

                    let mut vy_view = fields.index_axis_mut(ndarray::Axis(0), vy_idx);
                    vy_view
                        .slice_mut(slice)
                        .assign(&self.pstd_solver.fields.uy.slice(slice));

                    let mut vz_view = fields.index_axis_mut(ndarray::Axis(0), vz_idx);
                    vz_view
                        .slice_mut(slice)
                        .assign(&self.pstd_solver.fields.uz.slice(slice));
                }
                DomainType::FDTD => {
                    let mut p_view = fields.index_axis_mut(ndarray::Axis(0), p_idx);
                    let slice = s![
                        region.start.0..region.end.0,
                        region.start.1..region.end.1,
                        region.start.2..region.end.2
                    ];
                    p_view
                        .slice_mut(slice)
                        .assign(&self.fdtd_solver.fields.p.slice(slice));

                    let mut vx_view = fields.index_axis_mut(ndarray::Axis(0), vx_idx);
                    vx_view
                        .slice_mut(slice)
                        .assign(&self.fdtd_solver.fields.ux.slice(slice));

                    let mut vy_view = fields.index_axis_mut(ndarray::Axis(0), vy_idx);
                    vy_view
                        .slice_mut(slice)
                        .assign(&self.fdtd_solver.fields.uy.slice(slice));

                    let mut vz_view = fields.index_axis_mut(ndarray::Axis(0), vz_idx);
                    vz_view
                        .slice_mut(slice)
                        .assign(&self.fdtd_solver.fields.uz.slice(slice));
                }
                DomainType::Hybrid => {
                    self.apply_hybrid_region_blended(fields, region)?;
                }
            }
        }

        let coupling_start = Instant::now();
        self.apply_coupling(fields)?;
        self.metrics.coupling_time += coupling_start.elapsed();

        if self.config.validation.enable_validation {
            self.validate_solution(fields, t)?;
        }

        self.time_step += 1;
        debug!(
            "Hybrid solver step {} completed in {:?}",
            self.time_step,
            update_start.elapsed()
        );

        Ok(())
    }

    /// Apply hybrid processing to transition region
    fn apply_hybrid_region_blended(
        &mut self,
        fields: &mut Array4<f64>,
        region: &DomainRegion,
    ) -> KwaversResult<()> {
        use crate::domain::field::mapping::UnifiedFieldType;
        let p_idx = UnifiedFieldType::Pressure.index();
        let vx_idx = UnifiedFieldType::VelocityX.index();
        let uy_idx = UnifiedFieldType::VelocityY.index();
        let uz_idx = UnifiedFieldType::VelocityZ.index();

        const BLEND_WIDTH: usize = 5;

        let nx = region.end.0 - region.start.0;
        let ny = region.end.1 - region.start.1;
        let nz = region.end.2 - region.start.2;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let dist_from_boundary = ((i.min(nx - i - 1))
                        .min(j.min(ny - j - 1))
                        .min(k.min(nz - k - 1)))
                        as f64;

                    let weight = if dist_from_boundary < BLEND_WIDTH as f64 {
                        0.5 * (1.0
                            + (std::f64::consts::PI * dist_from_boundary / BLEND_WIDTH as f64)
                                .cos())
                    } else {
                        1.0
                    };

                    let gi = region.start.0 + i;
                    let gj = region.start.1 + j;
                    let gk = region.start.2 + k;

                    fields[[p_idx, gi, gj, gk]] = weight * self.pstd_solver.fields.p[[gi, gj, gk]]
                        + (1.0 - weight) * self.fdtd_solver.fields.p[[gi, gj, gk]];

                    fields[[vx_idx, gi, gj, gk]] = weight
                        * self.pstd_solver.fields.ux[[gi, gj, gk]]
                        + (1.0 - weight) * self.fdtd_solver.fields.ux[[gi, gj, gk]];

                    fields[[uy_idx, gi, gj, gk]] = weight
                        * self.pstd_solver.fields.uy[[gi, gj, gk]]
                        + (1.0 - weight) * self.fdtd_solver.fields.uy[[gi, gj, gk]];

                    fields[[uz_idx, gi, gj, gk]] = weight
                        * self.pstd_solver.fields.uz[[gi, gj, gk]]
                        + (1.0 - weight) * self.fdtd_solver.fields.uz[[gi, gj, gk]];
                }
            }
        }

        debug!("Applied hybrid blending to region {:?}", region);
        Ok(())
    }

    /// Apply coupling between regions
    fn apply_coupling(&mut self, fields: &mut Array4<f64>) -> KwaversResult<()> {
        self.coupling
            .apply_coupling(fields, &self.regions, &self.grid)
    }

    /// Update domain decomposition based on current fields
    fn update_decomposition(
        &mut self,
        fields: &Array4<f64>,
        medium: &dyn Medium,
    ) -> KwaversResult<()> {
        let start = Instant::now();

        self.selector.update_metrics(fields);

        let new_regions = self.decomposer.decompose(
            &self.grid,
            medium,
            self.config.decomposition_strategy.clone(),
        )?;

        if new_regions.len() != self.regions.len() {
            use log::info;
            info!(
                "Domain decomposition updated: {} regions",
                new_regions.len()
            );
            self.regions = new_regions;
        }

        self.metrics.decomposition_time += start.elapsed();
        Ok(())
    }

    /// Validate solution quality
    fn validate_solution(&mut self, fields: &Array4<f64>, _time: f64) -> KwaversResult<()> {
        use crate::domain::field::mapping::UnifiedFieldType;

        let pressure = fields.index_axis(ndarray::Axis(0), UnifiedFieldType::Pressure.index());
        let has_nan = pressure.iter().any(|&x| x.is_nan());
        let has_inf = pressure.iter().any(|&x| x.is_infinite());

        if has_nan || has_inf {
            self.validation_results.quality_score = 0.0;
            self.validation_results.nan_inf_count += 1;

            if self.config.validation.check_nan_inf {
                return Err(KwaversError::Validation(
                    crate::core::error::ValidationError::FieldValidation {
                        field: "pressure".to_string(),
                        value: format!("NaN: {has_nan}, Inf: {has_inf}"),
                        constraint: "Must be finite".to_string(),
                    },
                ));
            }
        }

        Ok(())
    }

    /// Get performance metrics
    pub fn metrics(&self) -> &HybridMetrics {
        &self.metrics
    }

    /// Get validation results
    pub fn validation_results(&self) -> &ValidationResults {
        &self.validation_results
    }

    /// Extract recorded sensor data from the internal FDTD solver.
    /// Returns None if no sensors are configured or no data has been recorded.
    pub fn extract_recorded_sensor_data(&self) -> Option<ndarray::Array2<f64>> {
        self.fdtd_solver.extract_recorded_sensor_data()
    }
}
