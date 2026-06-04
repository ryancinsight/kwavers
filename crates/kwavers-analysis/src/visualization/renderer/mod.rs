//! 3D Renderer - GPU-Accelerated Volume Rendering
//!
//! This module implements high-performance 3D rendering for scientific visualization.

pub mod gpu;
pub mod isosurface;
pub mod pipeline;
pub mod uniforms;
pub mod volume;

pub use gpu::RendererGpuContext;
pub use isosurface::IsosurfaceExtractor;
pub use pipeline::{ComputePipeline, RenderPipeline};
pub use uniforms::VolumeUniforms;
pub use volume::VolumeRenderer;

use crate::visualization::{RenderQuality, VisualizationConfig};
use kwavers_core::error::KwaversResult;
use kwavers_field::UnifiedFieldType;
use kwavers_grid::Grid;
use ndarray::Array3;

/// Main 3D renderer orchestrator
#[derive(Debug)]
pub struct Renderer3D {
    config: VisualizationConfig,
    volume: VolumeRenderer,
    isosurface: IsosurfaceExtractor,
    gpu: Option<RendererGpuContext>,
}

impl Renderer3D {
    /// Create a new 3D renderer
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: VisualizationConfig) -> KwaversResult<Self> {
        let volume = VolumeRenderer::new(&config)?;
        let isosurface = IsosurfaceExtractor::new(&config)?;

        let gpu = if config.gpu_enabled {
            Some(RendererGpuContext::new(&config)?)
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn render_field(
        &mut self,
        field: &Array3<f64>,
        field_type: UnifiedFieldType,
        grid: &Grid,
    ) -> KwaversResult<Vec<u8>> {
        match self.config.render_quality {
            RenderQuality::Draft => self.volume.render_draft(field, field_type, grid),
            RenderQuality::Low => self.volume.render_draft(field, field_type, grid),
            RenderQuality::Medium => self.volume.render_production(field, field_type, grid),
            RenderQuality::High => self.volume.render_production(field, field_type, grid),
            RenderQuality::Production => self.volume.render_production(field, field_type, grid),
            RenderQuality::Publication => self.volume.render_publication(field, field_type, grid),
        }
    }

    /// Extract isosurface
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn extract_isosurface(
        &mut self,
        field: &Array3<f64>,
        threshold: f64,
    ) -> KwaversResult<Vec<[f32; 3]>> {
        self.isosurface.extract(field, threshold)
    }

    /// Get memory usage
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn memory_usage(&self) -> usize {
        self.volume.memory_usage()
            + self.isosurface.memory_usage()
            + self.gpu.as_ref().map_or(0, |g| g.memory_usage())
    }

    /// Create a new renderer (alias for new)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn create(config: VisualizationConfig) -> KwaversResult<Self> {
        Self::new(config)
    }
    /// Render volume.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub async fn render_volume(
        &mut self,
        field: &Array3<f64>,
        field_type: UnifiedFieldType,
        grid: &Grid,
    ) -> KwaversResult<Vec<u8>> {
        self.render_field(field, field_type, grid)
    }
    /// Render multi volume.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub async fn render_multi_volume(
        &mut self,
        fields: Vec<(UnifiedFieldType, &Array3<f64>)>,
        grid: &Grid,
    ) -> KwaversResult<Vec<u8>> {
        let mut iter = fields.into_iter();
        let (first_type, first_field) = iter.next().ok_or_else(|| {
            kwavers_core::error::KwaversError::InvalidInput("no fields".to_string())
        })?;

        let mut out = self.render_field(first_field, first_type, grid)?;
        if !self.config.enable_transparency {
            return Ok(out);
        }

        for (field_type, field) in iter {
            let src = self.render_field(field, field_type, grid)?;
            alpha_over_in_place(&mut out, &src)?;
        }

        Ok(out)
    }
}

fn alpha_over_in_place(dst: &mut [u8], src: &[u8]) -> KwaversResult<()> {
    if dst.len() != src.len() {
        return Err(kwavers_core::error::KwaversError::InvalidInput(
            "image size mismatch".to_string(),
        ));
    }
    for (d_px, s_px) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
        let sa = (s_px[3] as f32) / 255.0;
        let da = (d_px[3] as f32) / 255.0;
        let out_a = sa + da * (1.0 - sa);
        if out_a <= 0.0 {
            d_px[0] = 0;
            d_px[1] = 0;
            d_px[2] = 0;
            d_px[3] = 0;
            continue;
        }
        for c in 0..3 {
            let sc = (s_px[c] as f32) / 255.0;
            let dc = (d_px[c] as f32) / 255.0;
            let out_c = (sc * sa + dc * da * (1.0 - sa)) / out_a;
            d_px[c] = (out_c.clamp(0.0, 1.0) * 255.0) as u8;
        }
        d_px[3] = (out_a.clamp(0.0, 1.0) * 255.0) as u8;
    }
    Ok(())
}
