//! 3D Renderer - GPU-Accelerated Volume Rendering
//!
//! This module implements high-performance 3D rendering for scientific visualization.

pub mod isosurface;
pub mod volume;

pub use isosurface::IsosurfaceExtractor;
pub use volume::VolumeRenderer;

use crate::visualization::{RenderQuality, VisualizationConfig};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_field::UnifiedFieldType;
use kwavers_grid::Grid;
use leto::Array3;

/// Main 3D renderer orchestrator
#[derive(Debug)]
pub struct Renderer3D {
    config: VisualizationConfig,
    volume: VolumeRenderer,
    isosurface: IsosurfaceExtractor,
}

impl Renderer3D {
    /// Create a new 3D renderer
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn new(config: VisualizationConfig) -> KwaversResult<Self> {
        let volume = VolumeRenderer::new(&config)?;
        let isosurface = IsosurfaceExtractor::new(&config)?;

        Ok(Self {
            config,
            volume,
            isosurface,
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
        self.volume.memory_usage() + self.isosurface.memory_usage()
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
    /// Render all supplied volumes in order.
    ///
    /// A single field renders under every configuration. Multiple fields are
    /// alpha-composited in input order and therefore require transparency.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] when `fields` is empty or when
    ///   multiple fields are supplied while transparency is disabled.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub async fn render_multi_volume(
        &mut self,
        fields: Vec<(UnifiedFieldType, &Array3<f64>)>,
        grid: &Grid,
    ) -> KwaversResult<Vec<u8>> {
        let mut iter = fields.into_iter();
        let (first_type, first_field) = iter
            .next()
            .ok_or_else(|| KwaversError::InvalidInput("no fields".to_string()))?;

        if !self.config.enable_transparency {
            if iter.next().is_some() {
                return Err(KwaversError::InvalidInput(
                    "multi-field rendering requires transparency for compositing".to_string(),
                ));
            }
            return self.render_field(first_field, first_type, grid);
        }

        let mut out = self.render_field(first_field, first_type, grid)?;
        for (field_type, field) in iter {
            let src = self.render_field(field, field_type, grid)?;
            alpha_over_in_place(&mut out, &src)?;
        }

        Ok(out)
    }
}

fn alpha_over_in_place(dst: &mut [u8], src: &[u8]) -> KwaversResult<()> {
    if dst.len() != src.len() {
        return Err(KwaversError::InvalidInput(
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

#[cfg(test)]
mod tests {
    use super::Renderer3D;
    use crate::visualization::VisualizationConfig;
    use kwavers_field::UnifiedFieldType;
    use kwavers_grid::Grid;
    use leto::Array3;

    #[test]
    fn multi_volume_composites_each_field() {
        let mut config = VisualizationConfig::quality();
        config.gpu_enabled = false;
        let mut renderer = Renderer3D::new(config).expect("test renderer has no GPU requirement");
        let grid = Grid::new(2, 2, 2, 1.0, 1.0, 1.0).expect("test grid is valid");
        let mut first = Array3::zeros((2, 2, 2));
        let mut second = Array3::zeros((2, 2, 2));
        *first
            .get_mut([0, 0, 0])
            .expect("first field index is in bounds") = 1.0;
        *second
            .get_mut([1, 1, 1])
            .expect("second field index is in bounds") = 1.0;

        let first_image = pollster::block_on(
            renderer.render_multi_volume(vec![(UnifiedFieldType::Pressure, &first)], &grid),
        )
        .expect("single-field rendering succeeds");
        let composite_image = pollster::block_on(renderer.render_multi_volume(
            vec![
                (UnifiedFieldType::Pressure, &first),
                (UnifiedFieldType::Temperature, &second),
            ],
            &grid,
        ))
        .expect("multi-field compositing succeeds");

        assert_ne!(composite_image, first_image);
    }

    #[test]
    fn multi_volume_rejects_multiple_fields_without_transparency() {
        let mut renderer = Renderer3D::new(VisualizationConfig::performance())
            .expect("performance renderer has no GPU requirement");
        let grid = Grid::new(2, 2, 2, 1.0, 1.0, 1.0).expect("test grid is valid");
        let first = Array3::zeros((2, 2, 2));
        let second = Array3::zeros((2, 2, 2));

        let result = pollster::block_on(renderer.render_multi_volume(
            vec![
                (UnifiedFieldType::Pressure, &first),
                (UnifiedFieldType::Temperature, &second),
            ],
            &grid,
        ));

        assert!(matches!(
            result,
            Err(kwavers_core::error::KwaversError::InvalidInput(message))
                if message.contains("requires transparency")
        ));
    }
}
