//! Main coupling interface implementation

use super::{ConservationEnforcer, QualityMonitor, TransferOperators};
use super::{InterfaceGeometry, InterpolationManager, InterpolationScheme};
use crate::core::error::{ConfigError, KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use crate::solver::hybrid::domain_decomposition::DomainRegion;
use ndarray::{s, Array3, Array4};

/// Interface coupling data for a single boundary
#[derive(Debug, Clone)]
pub struct InterfaceCoupling {
    /// Source domain information
    pub source_domain: super::DomainInfo,
    /// Target domain information  
    pub target_domain: super::DomainInfo,
    /// Interface geometry
    pub interface_geometry: InterfaceGeometry,
    /// Transfer operators
    pub transfer_operators: TransferOperators,
    /// Quality metrics
    pub quality_metrics: super::InterfaceQualityMetrics,
}

/// Main coupling interface between PSTD and FDTD domains
#[derive(Debug)]
pub struct CouplingInterface {
    /// Interface geometry
    geometry: InterfaceGeometry,
    /// Interpolation manager
    interpolation: InterpolationManager,
    /// Conservation enforcer
    conservation: ConservationEnforcer,
    /// Quality monitor
    quality: QualityMonitor,
    /// Transfer operators
    transfer: TransferOperators,
}

impl CouplingInterface {
    /// Apply coupling between domains
    pub fn apply_coupling(
        &mut self,
        fields: &mut Array4<f64>,
        regions: &[DomainRegion],
        _grid: &Grid,
    ) -> KwaversResult<()> {
        // Extract field data from source region
        let source_region = regions.first().ok_or_else(|| {
            KwaversError::Config(ConfigError::MissingParameter {
                parameter: "source_region".to_string(),
                section: "coupling".to_string(),
            })
        })?;

        // Extract the interface data
        let interface_data = self.extract_interface_data(fields, source_region)?;

        // Generate interface coordinates
        let source_coords = self.get_interface_coords(true)?;
        let target_coords = self.get_interface_coords(false)?;

        // Apply interpolation
        let interpolated =
            self.interpolation
                .interpolate(&interface_data, &source_coords, &target_coords)?;

        // Create target field for conservation
        let target_field = if regions.len() > 1 {
            self.extract_interface_data(fields, &regions[1])?
        } else {
            interface_data.clone()
        };

        // Enforce conservation laws
        let conserved = self.conservation.enforce(&interpolated, &target_field)?;

        // Transfer to target region
        if regions.len() > 1 {
            let target_region = &regions[1];
            self.apply_to_target(fields, &conserved, target_region)?;
        }

        // Update quality metrics with current time
        let time = 0.0; // Would get from simulation context
        self.quality.update(&interface_data, &conserved, time);

        Ok(())
    }

    /// Extract interface data from fields
    fn extract_interface_data(
        &self,
        fields: &Array4<f64>,
        region: &DomainRegion,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz, _) = fields.dim();

        // Extract the region bounds
        let (sx, sy, sz) = region.start;
        let (ex, ey, ez) = region.end;

        // Create interface data for the region
        let mut interface_data = Array3::zeros((ex - sx, ey - sy, ez - sz));

        // Extract based on interface normal direction
        match self.geometry.normal_direction {
            0 => {
                // X-normal interface - extract YZ plane
                if sx < nx {
                    interface_data.assign(&fields.slice(s![sx, sy..ey, sz..ez, 0]));
                }
            }
            1 => {
                // Y-normal interface - extract XZ plane
                if sy < ny {
                    interface_data.assign(&fields.slice(s![sx..ex, sy, sz..ez, 0]));
                }
            }
            2 => {
                // Z-normal interface - extract XY plane
                if sz < nz {
                    interface_data.assign(&fields.slice(s![sx..ex, sy..ey, sz, 0]));
                }
            }
            _ => {
                return Err(KwaversError::Config(ConfigError::InvalidValue {
                    parameter: "normal_direction".to_string(),
                    value: self.geometry.normal_direction.to_string(),
                    constraint: "Must be 0, 1, or 2".to_string(),
                }))
            }
        }

        Ok(interface_data)
    }

    /// Apply conserved data to target region
    fn apply_to_target(
        &self,
        fields: &mut Array4<f64>,
        data: &Array3<f64>,
        region: &DomainRegion,
    ) -> KwaversResult<()> {
        let (nx, ny, nz, _) = fields.dim();

        // Extract the region bounds
        let (sx, sy, sz) = region.start;
        let (ex, ey, ez) = region.end;

        // Apply based on interface normal direction
        match self.geometry.normal_direction {
            0 => {
                // X-normal interface - apply to YZ plane
                if sx < nx && sx < ex {
                    let slice_data = data.slice(s![.., .., ..]);
                    fields
                        .slice_mut(s![sx, sy..ey.min(ny), sz..ez.min(nz), 0])
                        .assign(&slice_data);
                }
            }
            1 => {
                // Y-normal interface - apply to XZ plane
                if sy < ny && sy < ey {
                    let slice_data = data.slice(s![.., .., ..]);
                    fields
                        .slice_mut(s![sx..ex.min(nx), sy, sz..ez.min(nz), 0])
                        .assign(&slice_data);
                }
            }
            2 => {
                // Z-normal interface - apply to XY plane
                if sz < nz && sz < ez {
                    let slice_data = data.slice(s![.., .., ..]);
                    fields
                        .slice_mut(s![sx..ex.min(nx), sy..ey.min(ny), sz, 0])
                        .assign(&slice_data);
                }
            }
            _ => {
                return Err(KwaversError::Config(ConfigError::InvalidValue {
                    parameter: "normal_direction".to_string(),
                    value: self.geometry.normal_direction.to_string(),
                    constraint: "Must be 0, 1, or 2".to_string(),
                }))
            }
        }

        Ok(())
    }
    /// Create a new coupling interface
    pub fn new(
        source_grid: &Grid,
        target_grid: &Grid,
        scheme: InterpolationScheme,
    ) -> KwaversResult<Self> {
        let geometry = InterfaceGeometry::from_grids(source_grid, target_grid)?;
        let interpolation = InterpolationManager::new(scheme);
        let conservation = ConservationEnforcer::new(&geometry);
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
    pub fn transfer_fields(
        &mut self,
        source_fields: &Array3<f64>,
        target_fields: &mut Array3<f64>,
        t: f64,
    ) -> KwaversResult<()> {
        // Get source and target coordinates at interface
        let source_coords = self.get_interface_coords(true)?;
        let target_coords = self.get_interface_coords(false)?;

        // Interpolate fields
        let interpolated =
            self.interpolation
                .interpolate(source_fields, &source_coords, &target_coords)?;

        // Enforce conservation
        let conserved = self.conservation.enforce(&interpolated, target_fields)?;

        // Apply transfer
        self.transfer.apply(&conserved, target_fields)?;

        // Update quality metrics
        self.quality.update(&conserved, target_fields, t);

        Ok(())
    }

    /// Get interface coordinates
    fn get_interface_coords(&self, _source: bool) -> KwaversResult<Vec<(f64, f64, f64)>> {
        let mut coords = Vec::with_capacity(self.geometry.num_points);
        let plane_pos = self.geometry.plane_position;
        let (extent_1, extent_2) = self.geometry.extent;

        // Generate a grid of points on the interface plane
        let grid_size = (self.geometry.num_points as f64).sqrt() as usize;
        let step_1 = extent_1 / grid_size as f64;
        let step_2 = extent_2 / grid_size as f64;

        match self.geometry.normal_direction {
            0 => {
                // X-normal interface (YZ plane)
                for j in 0..grid_size {
                    for k in 0..grid_size {
                        coords.push((plane_pos, j as f64 * step_1, k as f64 * step_2));
                    }
                }
            }
            1 => {
                // Y-normal interface (XZ plane)
                for i in 0..grid_size {
                    for k in 0..grid_size {
                        coords.push((i as f64 * step_1, plane_pos, k as f64 * step_2));
                    }
                }
            }
            2 => {
                // Z-normal interface (XY plane)
                for i in 0..grid_size {
                    for j in 0..grid_size {
                        coords.push((i as f64 * step_1, j as f64 * step_2, plane_pos));
                    }
                }
            }
            _ => {
                return Err(KwaversError::Config(ConfigError::InvalidValue {
                    parameter: "normal_direction".to_string(),
                    value: self.geometry.normal_direction.to_string(),
                    constraint: "Must be 0, 1, or 2".to_string(),
                }))
            }
        }

        Ok(coords)
    }

    /// Get quality metrics
    #[must_use]
    pub fn quality_metrics(&self) -> super::InterfaceQualityMetrics {
        self.quality.get_metrics()
    }
}
