//! Coupling interface module for hybrid PSTD/FDTD solver
//!
//! This module provides the coupling infrastructure between different
//! computational domains, ensuring conservation laws and numerical stability.

mod geometry;
mod interpolation;
mod conservation;
mod synchronization;
mod metrics;
mod transfer;

pub use geometry::{InterfaceGeometry, DomainInfo};
pub use interpolation::{InterpolationScheme, Interpolator};
pub use conservation::{ConservationValidator, ConservationMetrics};
pub use synchronization::{TimeSynchronizer, SyncStrategy};
pub use metrics::{InterfaceQualityMetrics, QualityValidator};
pub use transfer::{TransferOperators, FieldTransfer};

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::Array3;

/// Main coupling interface manager
pub struct CouplingInterface {
    geometry: InterfaceGeometry,
    interpolator: Box<dyn Interpolator>,
    conservation: ConservationValidator,
    synchronizer: TimeSynchronizer,
    quality_validator: QualityValidator,
    transfer: FieldTransfer,
}

impl CouplingInterface {
    /// Create a new coupling interface
    pub fn new(
        source_grid: &Grid,
        target_grid: &Grid,
        scheme: InterpolationScheme,
    ) -> KwaversResult<Self> {
        let geometry = InterfaceGeometry::from_grids(source_grid, target_grid)?;
        let interpolator = interpolation::create_interpolator(scheme);
        let conservation = ConservationValidator::new(&geometry);
        let synchronizer = TimeSynchronizer::new(source_grid.dt, target_grid.dt);
        let quality_validator = QualityValidator::new();
        let transfer = FieldTransfer::new(&geometry);
        
        Ok(Self {
            geometry,
            interpolator,
            conservation,
            synchronizer,
            quality_validator,
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
        // Synchronize time steps
        self.synchronizer.sync_time(t)?;
        
        // Interpolate fields
        let interpolated = self.interpolator.interpolate(
            source_fields,
            &self.geometry,
        )?;
        
        // Validate conservation
        self.conservation.validate(&interpolated, target_fields)?;
        
        // Transfer fields
        self.transfer.apply(&interpolated, target_fields)?;
        
        // Update quality metrics
        self.quality_validator.update_metrics(&interpolated, target_fields);
        
        Ok(())
    }
    
    /// Get interface quality metrics
    pub fn quality_metrics(&self) -> &InterfaceQualityMetrics {
        self.quality_validator.metrics()
    }
}