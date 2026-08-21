//! Visualization Engine Module
//!
//! Core engine for managing visualization pipeline.

use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_field::UnifiedFieldType;
use kwavers_grid::Grid;
use leto::{Array3, Array4};
#[cfg(not(feature = "gpu-visualization"))]
use log::warn;
use log::{debug, info};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::{
    config::{RenderQuality, VisualizationConfig},
    metrics::{MetricsTracker, VisualizationMetrics},
};

#[cfg(feature = "gpu-visualization")]
use super::config::MILLISECONDS_PER_SECOND;

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
    /// Provider-neutral transfer provider awaiting pipeline construction
    #[cfg(feature = "gpu-visualization")]
    transfer_provider: Option<Box<dyn super::transfer_contract::VisualizationTransferProvider>>,
    /// Interactive controls
    #[cfg(feature = "gpu-visualization")]
    controls: Option<controls::InteractiveControls>,
}

impl VisualizationEngine {
    /// Create a new visualization engine
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn create(config: VisualizationConfig) -> KwaversResult<Self> {
        config.validate()?;

        info!(
            "Creating visualization engine with target FPS: {}",
            config.target_fps
        );

        let mut parameters = HashMap::new();
        parameters.insert("frequency".to_string(), MHZ_TO_HZ);
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
            transfer_provider: None,
            #[cfg(feature = "gpu-visualization")]
            controls: None,
        })
    }

    /// Initialize GPU resources
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    /// Inject a provider-neutral transfer provider.
    ///
    /// The engine cannot construct device-backed providers itself: concrete
    /// acquisition lives behind the provider boundary (for example
    /// `kwavers-gpu`'s Hephaestus-backed WGPU implementation), keeping the
    /// dependency direction acyclic. Callers able to construct a provider must
    /// inject it before [`Self::initialize_gpu`].
    #[cfg(feature = "gpu-visualization")]
    pub fn set_transfer_provider(
        &mut self,
        provider: Box<dyn super::transfer_contract::VisualizationTransferProvider>,
    ) {
        self.transfer_provider = Some(provider);
    }

    /// Initialize GPU resources
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::System`] with `FeatureNotAvailable` when no
    /// transfer provider has been injected via [`Self::set_transfer_provider`];
    /// the engine never silently degrades a requested GPU path to CPU
    /// execution.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub async fn initialize_gpu(&mut self) -> KwaversResult<()> {
        info!("Initializing GPU resources for visualization");

        #[cfg(feature = "gpu-visualization")]
        {
            let provider = self.transfer_provider.take().ok_or_else(|| {
                KwaversError::System(kwavers_core::error::SystemError::FeatureNotAvailable {
                    feature: "gpu-visualization".to_string(),
                    reason: "no transfer provider injected; call set_transfer_provider with a \
                             provider from the GPU boundary first"
                        .to_string(),
                })
            })?;

            // Initialize renderer (CPU rasterization path)
            self.renderer = Some(renderer::Renderer3D::create(self.config.clone())?);

            // Wrap the injected provider in the provider-generic pipeline
            self.data_pipeline = Some(data_pipeline::DataPipeline::new(provider));

            // Initialize interactive controls
            self.controls = Some(controls::InteractiveControls::create(&self.config)?);
        }

        info!("GPU visualization initialization complete");
        Ok(())
    }

    /// Render a single field with 3D visualization
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub async fn render_field(
        &mut self,
        field: &Array3<f64>,
        field_type: UnifiedFieldType,
        grid: &Grid,
    ) -> KwaversResult<()> {
        let _start_time = Instant::now();

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
                renderer.render_volume(field, field_type, grid).await?;
                let render_time =
                    render_start.elapsed().as_secs_f32() * MILLISECONDS_PER_SECOND as f32;

                // Update metrics
                self.metrics.update(render_time, transfer_time);

                debug!(
                    "Rendered {:?} field: {:.2}ms render, {:.2}ms transfer",
                    field_type, render_time, transfer_time
                );
            } else {
                return Err(KwaversError::System(
                    kwavers_core::error::SystemError::FeatureNotAvailable {
                        feature: "gpu-visualization".to_string(),
                        reason: "GPU renderer and data pipeline are not initialized; call initialize_gpu() first".to_string(),
                    },
                ));
            }
        }

        #[cfg(not(feature = "gpu-visualization"))]
        {
            warn!("GPU visualization not enabled. Using fallback renderer.");
            super::fallback::render_field(field, field_type, grid)?;
        }

        Ok(())
    }

    /// Render multiple fields simultaneously.
    ///
    /// The fourth axis of `fields` supplies the field count and `field_types`
    /// supplies its matching semantic type for each slice. GPU rendering
    /// composites every slice; the CPU path sends every slice through the
    /// fallback renderer.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] when no fields are supplied or
    ///   when the field-type count does not match the fourth-axis length.
    /// - Returns [`KwaversError::System`] when GPU visualization is enabled but
    ///   [`Self::initialize_gpu`] has not initialized its renderer and data
    ///   pipeline.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub async fn render_multi_field(
        &mut self,
        fields: &Array4<f64>,
        field_types: &[UnifiedFieldType],
        grid: &Grid,
    ) -> KwaversResult<()> {
        let field_count = fields.shape()[3];
        if field_count == 0 {
            return Err(KwaversError::InvalidInput(
                "multi-field rendering requires at least one field".to_string(),
            ));
        }
        if field_types.len() != field_count {
            return Err(KwaversError::InvalidInput(format!(
                "multi-field rendering received {field_count} fields but {} field types",
                field_types.len()
            )));
        }

        #[cfg(feature = "gpu-visualization")]
        {
            if let (Some(renderer), Some(pipeline)) = (&mut self.renderer, &mut self.data_pipeline)
            {
                // Upload all fields to GPU
                let transfer_start = Instant::now();
                let mut contiguous_fields = Vec::with_capacity(field_count);
                for (i, &field_type) in field_types.iter().enumerate() {
                    let field = fields.index_axis::<3>(3, i)?.to_contiguous();
                    pipeline.upload_field(&field, field_type).await?;
                    contiguous_fields.push(field);
                }
                let transfer_time =
                    transfer_start.elapsed().as_secs_f32() * MILLISECONDS_PER_SECOND as f32;

                // Render all fields with transparency blending
                let render_start = Instant::now();
                let render_fields = field_types
                    .iter()
                    .copied()
                    .zip(contiguous_fields.iter())
                    .collect();
                renderer.render_multi_volume(render_fields, grid).await?;
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
            } else {
                return Err(KwaversError::System(
                    kwavers_core::error::SystemError::FeatureNotAvailable {
                        feature: "gpu-visualization".to_string(),
                        reason: "GPU renderer and data pipeline are not initialized; call initialize_gpu() first".to_string(),
                    },
                ));
            }
        }

        #[cfg(not(feature = "gpu-visualization"))]
        {
            warn!("GPU visualization not enabled. Using fallback renderer for all fields.");
            for (i, &field_type) in field_types.iter().enumerate() {
                let field = fields.index_axis::<3>(3, i)?.to_contiguous();
                super::fallback::render_field(&field, field_type, grid)?;
            }
        }

        Ok(())
    }

    /// Update a visualization parameter
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn update_parameter(&mut self, name: &str, value: f64) -> KwaversResult<()> {
        let mut params = self.parameters.lock().map_err(|_| {
            KwaversError::System(kwavers_core::error::SystemError::ResourceExhausted {
                resource: "Visualization parameters mutex".to_string(),
                reason: "Mutex poisoned".to_string(),
            })
        })?;
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
                RenderQuality::Publication => RenderQuality::Production,
                RenderQuality::Production => RenderQuality::High,
                RenderQuality::High => RenderQuality::Medium,
                RenderQuality::Medium => RenderQuality::Low,
                RenderQuality::Low => RenderQuality::Low,
                RenderQuality::Draft => RenderQuality::Draft,
            };
            debug!("Downgraded render quality to {:?}", self.config.quality);
        } else if current_fps > target_fps * 1.2 {
            // Upgrade quality if performance is good
            self.config.quality = match self.config.quality {
                RenderQuality::Draft => RenderQuality::Low,
                RenderQuality::Low => RenderQuality::Medium,
                RenderQuality::Medium => RenderQuality::High,
                RenderQuality::High => RenderQuality::Production,
                RenderQuality::Production => RenderQuality::Publication,
                RenderQuality::Publication => RenderQuality::Publication,
            };
            debug!("Upgraded render quality to {:?}", self.config.quality);
        }
    }

    /// Export visualization to file
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub async fn export(
        &self,
        field: &Array3<f64>,
        field_type: UnifiedFieldType,
        filename: &str,
    ) -> KwaversResult<()> {
        info!("Exporting visualization to {}", filename);

        #[cfg(feature = "gpu-visualization")]
        {
            super::fallback::export_field(field, field_type, filename)?;
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
