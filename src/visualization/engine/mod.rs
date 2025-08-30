//! Visualization Engine Module
//!
//! Core engine for managing visualization pipeline.

use crate::{
    error::KwaversResult, grid::Grid, physics::field_mapping::UnifiedFieldType as FieldType,
};
use log::{debug, info, warn};
use ndarray::{Array3, Array4};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::{
    config::{RenderQuality, VisualizationConfig, MILLISECONDS_PER_SECOND},
    metrics::{MetricsTracker, VisualizationMetrics},
};

#[cfg(feature = "gpu-visualization")]
use super::{controls, data_pipeline, renderer};

/// Main visualization engine
#[derive(Debug)]
pub struct VisualizationEngine {
    /// Configuration settings
    config: VisualizationConfig,
    /// Performance metrics tracker
    metrics: MetricsTracker,
    /// Render parameters (for dynamic adjustment)
    parameters: Arc<Mutex<HashMap<String, f64>>>,
    /// GPU renderer (if available)
    #[cfg(feature = "gpu-visualization")]
    renderer: Option<renderer::Renderer3D>,
    /// Data pipeline for GPU transfers
    #[cfg(feature = "gpu-visualization")]
    data_pipeline: Option<data_pipeline::DataPipeline>,
    /// Interactive controls
    #[cfg(feature = "gpu-visualization")]
    controls: Option<controls::InteractiveControls>,
}

impl VisualizationEngine {
    /// Create a new visualization engine
    pub fn create(config: VisualizationConfig) -> KwaversResult<Self> {
        config.validate()?;

        info!(
            "Creating visualization engine with target FPS: {}",
            config.target_fps
        );

        let mut parameters = HashMap::new();
        parameters.insert("frequency".to_string(), 1.0e6);
        parameters.insert("amplitude".to_string(), 1.0);
        parameters.insert("opacity".to_string(), 0.8);

        Ok(Self {
            config,
            metrics: MetricsTracker::new(),
            parameters: Arc::new(Mutex::new(parameters)),
            #[cfg(feature = "gpu-visualization")]
            renderer: None,
            #[cfg(feature = "gpu-visualization")]
            data_pipeline: None,
            #[cfg(feature = "gpu-visualization")]
            controls: None,
        })
    }

    /// Initialize GPU resources
    pub async fn initialize_gpu(&mut self) -> KwaversResult<()> {
        info!("Initializing GPU resources for visualization");

        #[cfg(feature = "gpu-visualization")]
        {
            // Create GPU context using wgpu
            let gpu_context = crate::gpu::backend::GpuBackend::new().await?;

            // Initialize renderer with GPU context
            self.renderer =
                Some(renderer::Renderer3D::create(&self.config, gpu_context.clone()).await?);

            // Initialize data pipeline for efficient GPU transfers
            self.data_pipeline = Some(data_pipeline::DataPipeline::new(gpu_context).await?);

            // Initialize interactive controls
            self.controls = Some(controls::InteractiveControls::create(&self.config)?);
        }

        info!("GPU visualization initialization complete");
        Ok(())
    }

    /// Render a single field with 3D visualization
    pub async fn render_field(
        &mut self,
        field: &Array3<f64>,
        field_type: FieldType,
        grid: &Grid,
    ) -> KwaversResult<()> {
        let start_time = Instant::now();

        #[cfg(feature = "gpu-visualization")]
        {
            if let (Some(renderer), Some(pipeline)) = (&mut self.renderer, &mut self.data_pipeline)
            {
                // Transfer field data to GPU
                let transfer_start = Instant::now();
                pipeline.upload_field(field, field_type).await?;
                let transfer_time =
                    transfer_start.elapsed().as_secs_f32() * MILLISECONDS_PER_SECOND as f32;

                // Render the field
                let render_start = Instant::now();
                renderer.render_volume(field_type, grid).await?;
                let render_time =
                    render_start.elapsed().as_secs_f32() * MILLISECONDS_PER_SECOND as f32;

                // Update metrics
                self.metrics.update(render_time, transfer_time);

                debug!(
                    "Rendered {} field: {:.2}ms render, {:.2}ms transfer",
                    format!("{:?}", field_type),
                    render_time,
                    transfer_time
                );
            } else {
                warn!("GPU visualization not initialized. Call initialize_gpu() first.");
                // Use fallback renderer
                super::fallback::render_field(field, field_type, grid)?;
            }
        }

        #[cfg(not(feature = "gpu-visualization"))]
        {
            warn!("GPU visualization not enabled. Using fallback renderer.");
            super::fallback::render_field(field, field_type, grid)?;
        }

        Ok(())
    }

    /// Render multiple fields simultaneously
    pub async fn render_multi_field(
        &mut self,
        fields: &Array4<f64>,
        field_types: &[FieldType],
        grid: &Grid,
    ) -> KwaversResult<()> {
        #[cfg(feature = "gpu-visualization")]
        {
            if let (Some(renderer), Some(pipeline)) = (&mut self.renderer, &mut self.data_pipeline)
            {
                // Upload all fields to GPU
                let transfer_start = Instant::now();
                for (i, &field_type) in field_types.iter().enumerate() {
                    if i < fields.shape()[3] {
                        let field = fields.slice(ndarray::s![.., .., .., i]);
                        pipeline.upload_field(&field.to_owned(), field_type).await?;
                    }
                }
                let transfer_time =
                    transfer_start.elapsed().as_secs_f32() * MILLISECONDS_PER_SECOND as f32;

                // Render all fields with transparency blending
                let render_start = Instant::now();
                renderer.render_multi_volume(field_types, grid).await?;
                let render_time =
                    render_start.elapsed().as_secs_f32() * MILLISECONDS_PER_SECOND as f32;

                // Update metrics
                self.metrics.update(render_time, transfer_time);

                info!(
                    "Rendered {} fields: {:.2}ms render, {:.2}ms transfer",
                    field_types.len(),
                    render_time,
                    transfer_time
                );
            }
        }

        #[cfg(not(feature = "gpu-visualization"))]
        {
            warn!("Multi-field rendering requires GPU visualization feature");
            // Render first field as fallback
            if !field_types.is_empty() && fields.shape()[3] > 0 {
                let field = fields.slice(ndarray::s![.., .., .., 0]).to_owned();
                super::fallback::render_field(&field, field_types[0], grid)?;
            }
        }

        Ok(())
    }

    /// Update a visualization parameter
    pub fn update_parameter(&mut self, name: &str, value: f64) -> KwaversResult<()> {
        let mut params = self.parameters.lock().unwrap();
        params.insert(name.to_string(), value);
        debug!("Updated parameter {} = {}", name, value);
        Ok(())
    }

    /// Get current performance metrics
    pub fn metrics(&self) -> &VisualizationMetrics {
        self.metrics.current()
    }

    /// Check if meeting performance targets
    pub fn meets_performance_targets(&self) -> bool {
        self.metrics.meets_target(self.config.target_fps)
    }

    /// Adjust quality based on performance
    pub fn auto_adjust_quality(&mut self) {
        if !self.config.enable_profiling {
            return;
        }

        let current_fps = self.metrics.current().fps;
        let target_fps = self.config.target_fps;

        if current_fps < target_fps * 0.8 {
            // Downgrade quality if performance is poor
            self.config.quality = match self.config.quality {
                RenderQuality::High => RenderQuality::Medium,
                RenderQuality::Medium => RenderQuality::Low,
                RenderQuality::Low => RenderQuality::Low,
            };
            debug!("Downgraded render quality to {:?}", self.config.quality);
        } else if current_fps > target_fps * 1.2 {
            // Upgrade quality if performance is good
            self.config.quality = match self.config.quality {
                RenderQuality::Low => RenderQuality::Medium,
                RenderQuality::Medium => RenderQuality::High,
                RenderQuality::High => RenderQuality::High,
            };
            debug!("Upgraded render quality to {:?}", self.config.quality);
        }
    }

    /// Export visualization to file
    pub async fn export(
        &self,
        field: &Array3<f64>,
        field_type: FieldType,
        filename: &str,
    ) -> KwaversResult<()> {
        info!("Exporting visualization to {}", filename);

        #[cfg(feature = "gpu-visualization")]
        {
            if let Some(renderer) = &self.renderer {
                renderer.export_frame(filename).await?;
            } else {
                super::fallback::export_field(field, field_type, filename)?;
            }
        }

        #[cfg(not(feature = "gpu-visualization"))]
        {
            super::fallback::export_field(field, field_type, filename)?;
        }

        Ok(())
    }

    /// Clean up GPU resources
    pub fn cleanup(&mut self) {
        #[cfg(feature = "gpu-visualization")]
        {
            self.renderer = None;
            self.data_pipeline = None;
            self.controls = None;
        }
        info!("Cleaned up visualization resources");
    }
}
