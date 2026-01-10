//! 3D Renderer - GPU-Accelerated Volume Rendering
//!
//! This module implements high-performance 3D rendering for scientific visualization.

pub mod gpu;
pub mod isosurface;
pub mod pipeline;
pub mod uniforms;
pub mod volume;

pub use gpu::GpuContext;
pub use isosurface::IsosurfaceExtractor;
pub use pipeline::{ComputePipeline, RenderPipeline};
pub use uniforms::VolumeUniforms;
pub use volume::VolumeRenderer;

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::visualization::{FieldType, VisualizationConfig};
use ndarray::Array3;

/// Main 3D renderer orchestrator
#[derive(Debug)]
pub struct Renderer3D {
    config: VisualizationConfig,
    volume: VolumeRenderer,
    isosurface: IsosurfaceExtractor,
    gpu: Option<GpuContext>,
}

impl Renderer3D {
    /// Create a new 3D renderer
    pub fn new(config: VisualizationConfig) -> KwaversResult<Self> {
        let volume = VolumeRenderer::new(&config)?;
        let isosurface = IsosurfaceExtractor::new(&config)?;

        let gpu = if config.gpu_enabled {
            Some(GpuContext::new(&config)?)
        } else {
            None
        };

        Ok(Self {
            config,
            volume,
            isosurface,
            gpu,
        })
    }

    /// Render a field
    pub fn render_field(
        &mut self,
        field: &Array3<f64>,
        field_type: FieldType,
        grid: &Grid,
    ) -> KwaversResult<Vec<u8>> {
        match self.config.render_quality {
            crate::visualization::RenderQuality::Draft => {
                self.volume.render_draft(field, field_type, grid)
            }
            crate::visualization::RenderQuality::Low => {
                self.volume.render_draft(field, field_type, grid)
            }
            crate::visualization::RenderQuality::Medium => {
                self.volume.render_production(field, field_type, grid)
            }
            crate::visualization::RenderQuality::High => {
                self.volume.render_production(field, field_type, grid)
            }
            crate::visualization::RenderQuality::Production => {
                self.volume.render_production(field, field_type, grid)
            }
            crate::visualization::RenderQuality::Publication => {
                self.volume.render_publication(field, field_type, grid)
            }
        }
    }

    /// Extract isosurface
    pub fn extract_isosurface(
        &mut self,
        field: &Array3<f64>,
        threshold: f64,
    ) -> KwaversResult<Vec<[f32; 3]>> {
        self.isosurface.extract(field, threshold)
    }

    /// Get memory usage
    pub fn memory_usage(&self) -> usize {
        self.volume.memory_usage()
            + self.isosurface.memory_usage()
            + self.gpu.as_ref().map_or(0, |g| g.memory_usage())
    }

    /// Create a new renderer (alias for new)
    pub fn create(config: VisualizationConfig) -> KwaversResult<Self> {
        Self::new(config)
    }

    /// Render volume data
    ///
    /// **Implementation Status**: Interface defined for volume rendering integration
    /// **Future**: Sprint 127+ will integrate with VolumeRenderer for MIP/ray marching
    /// **Current**: Returns empty buffer maintaining API contract for integration testing
    pub async fn render_volume(
        &mut self,
        _field_type: FieldType,
        _grid: &Grid,
    ) -> KwaversResult<Vec<u8>> {
        // API contract maintained for integration testing
        Ok(vec![])
    }

    /// Render multiple volume fields
    ///
    /// **Implementation Status**: Interface defined for multi-field visualization
    /// **Future**: Sprint 127+ will support composite rendering with multiple colormaps
    /// **Current**: Returns empty buffer maintaining API contract for integration testing
    pub async fn render_multi_volume(
        &mut self,
        _fields: Vec<(FieldType, &Array3<f64>)>,
        _grid: &Grid,
    ) -> KwaversResult<Vec<u8>> {
        // API contract maintained for integration testing
        Ok(vec![])
    }

    /// Export rendered frame
    ///
    /// **Implementation Status**: Interface defined for frame export
    /// **Future**: Sprint 127+ will support PNG/JPEG export via image crate
    /// Current: No-op maintaining API contract
    pub fn export_frame(&self, _path: &std::path::Path) -> KwaversResult<()> {
        // API contract maintained for integration testing
        Ok(())
    }
}
