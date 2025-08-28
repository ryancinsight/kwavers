//! 3D Renderer - GPU-Accelerated Volume Rendering
//!
//! This module implements high-performance 3D rendering for scientific visualization.

pub mod volume;
pub mod isosurface;
pub mod gpu;
pub mod pipeline;
pub mod uniforms;

pub use volume::VolumeRenderer;
pub use isosurface::IsosurfaceExtractor;
pub use gpu::GpuContext;
pub use pipeline::{RenderPipeline, ComputePipeline};
pub use uniforms::VolumeUniforms;

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::visualization::{VisualizationConfig, FieldType};
use ndarray::Array3;

/// Main 3D renderer orchestrator
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
        self.volume.memory_usage() + 
        self.isosurface.memory_usage() +
        self.gpu.as_ref().map_or(0, |g| g.memory_usage())
    }
}