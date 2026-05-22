//! CouplingInterface implementation

use crate::core::constants::thermodynamic::BODY_TEMPERATURE_C;
use crate::core::error::{ConfigError, KwaversError, KwaversResult};
use crate::domain::field::mapping::UnifiedFieldType;
use crate::domain::grid::Grid;
use crate::solver::forward::hybrid::coupling::{
    HybridCouplingConservationEnforcer, HybridInterpolationScheme, InterfaceGeometry,
    InterfaceQualityMetrics, InterpolationManager, QualityMonitor, TransferOperators,
};
use crate::solver::forward::hybrid::domain_decomposition::DomainRegion;
use ndarray::{s, Array3, Array4};

/// Main coupling interface between PSTD and FDTD domains
#[derive(Debug)]
pub struct CouplingInterface {
    geometry: InterfaceGeometry,
    interpolation: InterpolationManager,
    conservation: HybridCouplingConservationEnforcer,
    quality: QualityMonitor,
    transfer: TransferOperators,
}

impl CouplingInterface {
    /// Create a new coupling interface
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(
        source_grid: &Grid,
        target_grid: &Grid,
        scheme: HybridInterpolationScheme,
    ) -> KwaversResult<Self> {
        let geometry = InterfaceGeometry::from_grids(source_grid, target_grid)?;
        let interpolation = InterpolationManager::new(scheme);
        let conservation = HybridCouplingConservationEnforcer::new(&geometry);
        let quality = QualityMonitor::new();
        let transfer = TransferOperators::new(&geometry)?;

        Ok(Self {
            geometry,
            interpolation,
            conservation,
            quality,
            transfer,
        })
    }

    /// Transfer fields from source to target domain
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn transfer_fields(
        &mut self,
        source_fields: &Array3<f64>,
        target_fields: &mut Array3<f64>,
        t: f64,
    ) -> KwaversResult<()> {
        let source_coords = self.get_interface_coords(true)?;
        let target_coords = self.get_interface_coords(false)?;

        let interpolated =
            self.interpolation
                .interpolate(source_fields, &source_coords, &target_coords)?;

        let conserved = self.conservation.enforce(&interpolated, target_fields)?;

        self.transfer.apply(&conserved, target_fields)?;

        self.quality.update(&conserved, target_fields, t);

        Ok(())
    }

    /// Apply coupling between domains
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn apply_coupling(
        &mut self,
        fields: &mut Array4<f64>,
        regions: &[DomainRegion],
        _grid: &Grid,
    ) -> KwaversResult<()> {
        if regions.len() < 2 {
            return Ok(());
        }

        let source_region = regions.first().ok_or_else(|| {
            KwaversError::Config(ConfigError::MissingParameter {
                parameter: "source_region".to_owned(),
                section: "coupling".to_owned(),
            })
        })?;

        let interface_data = self.extract_interface_data(fields, source_region)?;

        let source_coords = self.get_interface_coords(true)?;
        let target_coords = self.get_interface_coords(false)?;

        let interpolated =
            self.interpolation
                .interpolate(&interface_data, &source_coords, &target_coords)?;

        // Restrict conservation to the physical interface plane. Region-shaped
        // buffers keep extraction/writing layout-stable, but inactive planes
        // are not transferred and must not receive affine conservation mass.
        let interpolated = self.extract_active_plane(&interpolated)?;

        let target_field = self.extract_interface_data(fields, &regions[1])?;
        let target_field = self.extract_active_plane(&target_field)?;

        let conserved = self.conservation.enforce(&interpolated, &target_field)?;

        let target_region = &regions[1];
        self.apply_to_target(fields, &conserved, target_region)?;

        // Conservation has already projected the interpolated trace into the
        // target interface integral/energy class, so diagnostics compare the
        // final transfer against the target trace that defined the constraint.
        let time = 0.0;
        self.quality.update(&conserved, &target_field, time);

        Ok(())
    }

    /// Get quality metrics
    #[must_use]
    pub fn quality_metrics(&self) -> InterfaceQualityMetrics {
        self.quality.get_metrics()
    }

    fn extract_interface_data(
        &self,
        fields: &Array4<f64>,
        region: &DomainRegion,
    ) -> KwaversResult<Array3<f64>> {
        let (n_fields, nx, ny, nz) = fields.dim();
        let p_idx = UnifiedFieldType::Pressure.index();
        if p_idx >= n_fields {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "field_layout".to_owned(),
                value: format!("{n_fields} components"),
                constraint: format!("at least {} component(s)", p_idx + 1),
            }));
        }

        let (sx, sy, sz) = region.start;
        let (ex, ey, ez) = region.end;
        let ex = ex.min(nx);
        let ey = ey.min(ny);
        let ez = ez.min(nz);

        // Store a region-shaped pressure buffer with only the coupling plane
        // populated. This preserves the existing interpolation/conservation
        // contract while enforcing the component-first field layout.
        let mut interface_data = Array3::zeros((ex - sx, ey - sy, ez - sz));
        let pressure = fields.index_axis(ndarray::Axis(0), p_idx);

        match self.geometry.normal_direction {
            0 => {
                if sx < nx {
                    interface_data
                        .slice_mut(s![0, .., ..])
                        .assign(&pressure.slice(s![sx, sy..ey, sz..ez]));
                }
            }
            1 => {
                if sy < ny {
                    interface_data
                        .slice_mut(s![.., 0, ..])
                        .assign(&pressure.slice(s![sx..ex, sy, sz..ez]));
                }
            }
            2 => {
                if sz < nz {
                    interface_data
                        .slice_mut(s![.., .., 0])
                        .assign(&pressure.slice(s![sx..ex, sy..ey, sz]));
                }
            }
            _ => {
                return Err(KwaversError::Config(ConfigError::InvalidValue {
                    parameter: "normal_direction".to_owned(),
                    value: self.geometry.normal_direction.to_string(),
                    constraint: "Must be 0, 1, or 2".to_owned(),
                }))
            }
        }

        Ok(interface_data)
    }

    fn apply_to_target(
        &self,
        fields: &mut Array4<f64>,
        data: &Array3<f64>,
        region: &DomainRegion,
    ) -> KwaversResult<()> {
        let (n_fields, nx, ny, nz) = fields.dim();
        let p_idx = UnifiedFieldType::Pressure.index();
        if p_idx >= n_fields {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "field_layout".to_owned(),
                value: format!("{n_fields} components"),
                constraint: format!("at least {} component(s)", p_idx + 1),
            }));
        }

        let (sx, sy, sz) = region.start;
        let (ex, ey, ez) = region.end;
        let ex = ex.min(nx);
        let ey = ey.min(ny);
        let ez = ez.min(nz);
        let mut pressure = fields.index_axis_mut(ndarray::Axis(0), p_idx);

        match self.geometry.normal_direction {
            0 => {
                if sx < nx && sx < ex {
                    pressure
                        .slice_mut(s![sx, sy..ey, sz..ez])
                        .assign(&data.slice(s![0, .., ..]));
                }
            }
            1 => {
                if sy < ny && sy < ey {
                    pressure
                        .slice_mut(s![sx..ex, sy, sz..ez])
                        .assign(&data.slice(s![.., 0, ..]));
                }
            }
            2 => {
                if sz < nz && sz < ez {
                    pressure
                        .slice_mut(s![sx..ex, sy..ey, sz])
                        .assign(&data.slice(s![.., .., 0]));
                }
            }
            _ => {
                return Err(KwaversError::Config(ConfigError::InvalidValue {
                    parameter: "normal_direction".to_owned(),
                    value: self.geometry.normal_direction.to_string(),
                    constraint: "Must be 0, 1, or 2".to_owned(),
                }))
            }
        }

        Ok(())
    }

    /// Return the active interface plane with the original axis order.
    ///
    /// ## Theorem
    ///
    /// For a region-shaped buffer `B`, `apply_to_target` writes only the
    /// codimension-one slice selected by `normal_direction`. Conservation over
    /// all of `B` is not conservative for the physical transfer: any correction
    /// placed outside that slice is discarded. Restricting the conservation
    /// domain to this active slice makes the measured and applied integral
    /// identical.
    ///
    /// # Errors
    /// - Returns [`KwaversError::Config`] if the active plane is empty or the
    ///   normal direction is invalid.
    fn extract_active_plane(&self, data: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = data.dim();
        match self.geometry.normal_direction {
            0 if nx > 0 => Ok(data.slice(s![0..1, .., ..]).to_owned()),
            1 if ny > 0 => Ok(data.slice(s![.., 0..1, ..]).to_owned()),
            2 if nz > 0 => Ok(data.slice(s![.., .., 0..1]).to_owned()),
            0 | 1 | 2 => Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "interface_plane".to_owned(),
                value: format!("{:?}", data.dim()),
                constraint: "active interface axis must be nonempty".to_owned(),
            })),
            _ => Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "normal_direction".to_owned(),
                value: self.geometry.normal_direction.to_string(),
                constraint: "Must be 0, 1, or 2".to_owned(),
            })),
        }
    }

    fn get_interface_coords(&self, _source: bool) -> KwaversResult<Vec<(f64, f64, f64)>> {
        let mut coords = Vec::with_capacity(self.geometry.num_points);
        let plane_pos = self.geometry.plane_position;
        let (extent_1, extent_2) = self.geometry.extent;

        let grid_size = (self.geometry.num_points as f64).sqrt() as usize;
        let step_1 = extent_1 / grid_size as f64;
        let step_2 = extent_2 / grid_size as f64;

        match self.geometry.normal_direction {
            0 => {
                for j in 0..grid_size {
                    for k in 0..grid_size {
                        coords.push((plane_pos, j as f64 * step_1, k as f64 * step_2));
                    }
                }
            }
            1 => {
                for i in 0..grid_size {
                    for k in 0..grid_size {
                        coords.push((i as f64 * step_1, plane_pos, k as f64 * step_2));
                    }
                }
            }
            2 => {
                for i in 0..grid_size {
                    for j in 0..grid_size {
                        coords.push((i as f64 * step_1, j as f64 * step_2, plane_pos));
                    }
                }
            }
            _ => {
                return Err(KwaversError::Config(ConfigError::InvalidValue {
                    parameter: "normal_direction".to_owned(),
                    value: self.geometry.normal_direction.to_string(),
                    constraint: "Must be 0, 1, or 2".to_owned(),
                }))
            }
        }

        Ok(coords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::thermodynamic::BODY_TEMPERATURE_C;
    use crate::solver::forward::hybrid::domain_decomposition::DomainType;
    use ndarray::Array4;

    fn test_interface() -> CouplingInterface {
        let grid = Grid::new(4, 3, 2, 1.0, 1.0, 1.0).unwrap();
        let mut interface =
            CouplingInterface::new(&grid, &grid, HybridInterpolationScheme::Linear).unwrap();
        interface.geometry.normal_direction = 0;
        interface
    }

    fn pressure_plane_sum(fields: &Array4<f64>, x: usize) -> f64 {
        let p_idx = UnifiedFieldType::Pressure.index();
        let mut sum = 0.0;
        for j in 0..3 {
            for k in 0..2 {
                sum += fields[[p_idx, x, j, k]];
            }
        }
        sum
    }

    #[test]
    fn extract_interface_data_reads_component_first_pressure_plane() {
        let interface = test_interface();
        let p_idx = UnifiedFieldType::Pressure.index();
        let t_idx = UnifiedFieldType::Temperature.index();
        let mut fields = Array4::<f64>::zeros((UnifiedFieldType::COUNT, 4, 3, 2));
        for i in 0..4 {
            for j in 0..3 {
                for k in 0..2 {
                    fields[[p_idx, i, j, k]] = 100.0 * i as f64 + 10.0 * j as f64 + k as f64;
                    fields[[t_idx, i, j, k]] = -10_000.0;
                }
            }
        }

        let region = DomainRegion::new((1, 0, 0), (4, 3, 2), DomainType::PSTD, 1.0);
        let data = interface.extract_interface_data(&fields, &region).unwrap();

        assert_eq!(data.dim(), (3, 3, 2));
        assert_eq!(data[[0, 2, 1]], fields[[p_idx, 1, 2, 1]]);
        assert_eq!(data[[1, 2, 1]], 0.0);
        assert_eq!(fields[[t_idx, 1, 2, 1]], -10_000.0);
    }

    #[test]
    fn apply_to_target_writes_component_first_pressure_plane_only() {
        let interface = test_interface();
        let p_idx = UnifiedFieldType::Pressure.index();
        let t_idx = UnifiedFieldType::Temperature.index();
        let mut fields = Array4::<f64>::zeros((UnifiedFieldType::COUNT, 4, 3, 2));
        fields.index_axis_mut(ndarray::Axis(0), t_idx).fill(25.0);
        let mut data = Array3::<f64>::zeros((2, 3, 2));
        for j in 0..3 {
            for k in 0..2 {
                data[[0, j, k]] = 10.0 * j as f64 + k as f64;
                data[[1, j, k]] = 999.0;
            }
        }

        let region = DomainRegion::new((2, 0, 0), (4, 3, 2), DomainType::FDTD, 1.0);
        interface
            .apply_to_target(&mut fields, &data, &region)
            .unwrap();

        for j in 0..3 {
            for k in 0..2 {
                assert_eq!(fields[[p_idx, 2, j, k]], data[[0, j, k]]);
                assert_eq!(fields[[p_idx, 3, j, k]], 0.0);
                assert_eq!(fields[[t_idx, 2, j, k]], 25.0);
            }
        }
    }

    #[test]
    fn apply_coupling_updates_target_plane_and_quality_against_target() {
        let mut interface = test_interface();
        let grid = Grid::new(4, 3, 2, 1.0, 1.0, 1.0).unwrap();
        let p_idx = UnifiedFieldType::Pressure.index();
        let t_idx = UnifiedFieldType::Temperature.index();
        let mut fields = Array4::<f64>::zeros((UnifiedFieldType::COUNT, 4, 3, 2));
        fields.index_axis_mut(ndarray::Axis(0), t_idx).fill(BODY_TEMPERATURE_C);

        let mut value = 1.0;
        for j in 0..3 {
            for k in 0..2 {
                fields[[p_idx, 0, j, k]] = value;
                value += 1.0;
                fields[[p_idx, 2, j, k]] = 20.0 + 2.0 * j as f64 + k as f64;
            }
        }

        let target_sum_before = pressure_plane_sum(&fields, 2);
        let regions = [
            DomainRegion::new((0, 0, 0), (2, 3, 2), DomainType::PSTD, 1.0),
            DomainRegion::new((2, 0, 0), (4, 3, 2), DomainType::FDTD, 1.0),
        ];

        interface
            .apply_coupling(&mut fields, &regions, &grid)
            .unwrap();

        let target_sum_after = pressure_plane_sum(&fields, 2);
        assert!((target_sum_after - target_sum_before).abs() < 1e-9);
        assert_eq!(fields[[t_idx, 2, 1, 1]], BODY_TEMPERATURE_C);
        assert_eq!(fields[[p_idx, 3, 1, 1]], 0.0);

        let metrics = interface.quality_metrics();
        assert!(metrics.conservation_error.abs() < 1e-9);
        assert!(metrics.interpolation_error.is_finite());
        assert_eq!(metrics.time, 0.0);
    }
}
