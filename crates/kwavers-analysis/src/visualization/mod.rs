//! # Visualization Module
//!
//! Core visualization and rendering infrastructure for the acoustic simulation library.
//!
//! ## Features
//! - **Real-time Rendering**: provider-backed field transfer and CPU rasterization
//! - **Selectable Backends**: Leto host storage or Hephaestus device transfer
//! - **Adaptive Quality**: Dynamic quality adjustment based on performance
//! - **Interactive Controls**: Real-time parameter adjustment and view manipulation
//! - **Data Export**: High-quality image and video export capabilities
//! - **Explicit CPU Mode**: CPU fallback is selected when GPU visualization is disabled
//!
//! ## Architecture
//! ```text
//! VisualizationEngine
//! ├── Config (configuration management)
//! ├── Metrics (performance tracking)
//! ├── Engine (core rendering pipeline)
//! ├── Transfer contract (provider-neutral field upload)
//! └── Fallback (CPU-based rendering when the feature is disabled)
//! ```
//!
//! ## Design Principles
//! - **SOLID**: Single responsibility per module
//! - **GRASP**: Modular organization under 500 lines
//! - **CUPID**: Composable visualization components
//! - **Zero-Cost**: Efficient GPU abstractions

// Module structure
pub mod config;
pub mod engine;
pub mod fallback;
pub mod metrics;

// Feature-gated visualization modules
#[cfg(feature = "gpu-visualization")]
pub mod controls;
#[cfg(feature = "gpu-visualization")]
pub mod data_pipeline;
#[cfg(feature = "gpu-visualization")]
pub mod renderer;
#[cfg(feature = "gpu-visualization")]
pub mod stream;
// Provider-neutral transfer seam; the domain crate owns the contract, the
// Hephaestus-backed WGPU implementation lives in kwavers-gpu.
#[cfg(feature = "gpu-visualization")]
pub mod transfer_contract;

// Re-exports for convenience
pub use config::{ColorScheme, RenderQuality, VisualizationConfig};
pub use engine::{UnconfiguredVisualizationProvider, VisualizationEngine};
pub use metrics::{MetricsTracker, VisualizationMetrics};
#[cfg(feature = "gpu-visualization")]
pub use transfer_contract::VisualizationTransferProvider;

// Re-export field types
pub use kwavers_field::mapping::UnifiedFieldType;

// GPU-specific re-exports
#[cfg(feature = "gpu-visualization")]
pub use controls::InteractiveControls;
#[cfg(feature = "gpu-visualization")]
pub use data_pipeline::{DataPipeline, TransferMode, TransferOptions};
#[cfg(feature = "gpu-visualization")]
pub use renderer::Renderer3D;
#[cfg(feature = "gpu-visualization")]
pub use stream::{FrameMetadata, VizFrame, VizStream};

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_grid::Grid;
    use leto::{Array3, Array4};

    #[cfg(feature = "gpu-visualization")]
    #[derive(Debug, Default)]
    struct RecordingProvider {
        last_transfer: Option<(UnifiedFieldType, Vec<f32>, TransferMode)>,
    }

    #[cfg(feature = "gpu-visualization")]
    impl VisualizationTransferProvider for RecordingProvider {
        fn device_name(&self) -> &str {
            "recording-provider"
        }

        fn is_available(&self) -> bool {
            true
        }

        fn transfer_field(
            &mut self,
            field_type: UnifiedFieldType,
            samples: &[f32],
            mode: TransferMode,
        ) -> kwavers_core::error::KwaversResult<()> {
            self.last_transfer = Some((field_type, samples.to_vec(), mode));
            Ok(())
        }

        fn memory_usage(&self) -> usize {
            self.last_transfer.as_ref().map_or(0, |(_, samples, _)| {
                samples.len() * std::mem::size_of::<f32>()
            })
        }
    }

    #[cfg(feature = "gpu-visualization")]
    fn create_configured_test_engine(
        config: VisualizationConfig,
    ) -> VisualizationEngine<RecordingProvider> {
        VisualizationEngine::create(config)
            .expect("valid visualization configuration")
            .set_transfer_provider(RecordingProvider::default())
    }

    fn create_test_grid() -> Grid {
        Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).expect("Failed to create test grid")
    }

    fn create_test_field() -> Array3<f64> {
        Array3::zeros((32, 32, 32))
    }

    #[test]
    fn test_visualization_config_default() {
        let config = VisualizationConfig::default();
        assert_eq!(config.target_fps, config::DEFAULT_TARGET_FPS);
        assert_eq!(config.quality, RenderQuality::Medium);
        assert_eq!(config.color_scheme, ColorScheme::Viridis);
        assert!(config.enable_transparency);
        assert_eq!(config.max_texture_size, config::DEFAULT_MAX_TEXTURE_SIZE);
        assert!(!config.enable_profiling);
    }

    #[test]
    fn test_visualization_config_performance() {
        let config = VisualizationConfig::performance();
        assert_eq!(config.quality, RenderQuality::Low);
        assert!(!config.enable_transparency);
        assert_eq!(config.max_texture_size, 256);
    }

    #[test]
    fn test_visualization_config_quality() {
        let config = VisualizationConfig::quality();
        assert_eq!(config.quality, RenderQuality::High);
        assert!(config.enable_transparency);
        assert_eq!(config.max_texture_size, 1024);
    }

    #[test]
    fn test_metrics_default() {
        let metrics = VisualizationMetrics::default();
        assert_eq!(metrics.fps, 0.0);
        assert_eq!(metrics.gpu_memory_usage, 0);
        assert_eq!(metrics.render_time_ms, 0.0);
        assert_eq!(metrics.transfer_time_ms, 0.0);
        assert_eq!(metrics.rendered_primitives, 0);
    }

    #[test]
    fn test_metrics_tracker() {
        let mut tracker = MetricsTracker::new();

        // Update with some measurements
        tracker.update(16.0, 2.0);
        tracker.update(15.0, 3.0);
        tracker.update(17.0, 2.5);

        let metrics = tracker.current();
        assert!(metrics.fps > 0.0);
        assert!(metrics.render_time_ms > 0.0);
        assert!(metrics.transfer_time_ms > 0.0);
    }

    #[test]
    fn test_performance_targets() {
        let mut tracker = MetricsTracker::new();

        // Simulate good performance
        for _ in 0..10 {
            tracker.update(10.0, 2.0); // 12ms total = ~83 FPS
        }

        assert!(tracker.meets_target(60.0)); // Should meet 60 FPS target
        assert!(!tracker.meets_target(100.0)); // Should not meet 100 FPS target
    }

    #[test]
    fn test_engine_creation() {
        let config = VisualizationConfig::default();
        let _engine = VisualizationEngine::create(config).unwrap();
    }

    #[test]
    fn test_parameter_update() {
        let config = VisualizationConfig::default();
        let mut engine = VisualizationEngine::create(config).unwrap();

        engine.update_parameter("frequency", 2.0e6).unwrap();
    }

    #[cfg(not(feature = "gpu-visualization"))]
    #[test]
    fn test_render_field_without_gpu() {
        let config = VisualizationConfig::default();
        let mut engine = VisualizationEngine::create(config).unwrap();

        let grid = create_test_grid();
        let field = create_test_field();

        engine
            .render_field(&field, UnifiedFieldType::Pressure, &grid)
            .unwrap();
    }

    #[cfg(not(feature = "gpu-visualization"))]
    #[test]
    fn test_render_multi_field_without_gpu() {
        let config = VisualizationConfig::default();
        let mut engine = VisualizationEngine::create(config).unwrap();

        let grid = create_test_grid();
        let fields = Array4::zeros((32, 32, 32, 3));
        let field_types = vec![
            UnifiedFieldType::Pressure,
            UnifiedFieldType::Temperature,
            UnifiedFieldType::LightFluence,
        ];

        engine
            .render_multi_field(&fields, &field_types, &grid)
            .unwrap();
    }

    #[cfg(feature = "gpu-visualization")]
    #[test]
    fn test_render_field_requires_gpu_initialization() {
        let config = VisualizationConfig::default();
        let mut engine = create_configured_test_engine(config);
        let grid = create_test_grid();
        let field = create_test_field();

        let result = engine.render_field(&field, UnifiedFieldType::Pressure, &grid);

        assert!(matches!(
            result,
            Err(kwavers_core::error::KwaversError::System(
                kwavers_core::error::SystemError::FeatureNotAvailable { feature, .. }
            )) if feature == "gpu-visualization"
        ));
    }

    #[cfg(feature = "gpu-visualization")]
    #[test]
    fn test_render_multi_field_requires_gpu_initialization() {
        let config = VisualizationConfig::default();
        let mut engine = create_configured_test_engine(config);
        let grid = create_test_grid();
        let fields = Array4::zeros((32, 32, 32, 2));
        let field_types = vec![UnifiedFieldType::Pressure, UnifiedFieldType::Temperature];

        let result = engine.render_multi_field(&fields, &field_types, &grid);

        assert!(matches!(
            result,
            Err(kwavers_core::error::KwaversError::System(
                kwavers_core::error::SystemError::FeatureNotAvailable { feature, .. }
            )) if feature == "gpu-visualization"
        ));
    }

    #[test]
    fn test_render_multi_field_rejects_field_type_mismatch() {
        let config = VisualizationConfig::default();
        #[cfg(feature = "gpu-visualization")]
        let mut engine = create_configured_test_engine(config);
        #[cfg(not(feature = "gpu-visualization"))]
        let mut engine = VisualizationEngine::create(config).unwrap();
        let grid = create_test_grid();
        let fields = Array4::zeros((32, 32, 32, 2));
        let field_types = vec![UnifiedFieldType::Pressure];

        let result = engine.render_multi_field(&fields, &field_types, &grid);

        assert!(matches!(
            result,
            Err(kwavers_core::error::KwaversError::InvalidInput(message))
                if message.contains("received 2 fields but 1 field types")
        ));
    }

    #[test]
    fn test_fallback_renderer() {
        let grid = create_test_grid();
        let field = create_test_field();

        fallback::render_field(&field, UnifiedFieldType::Pressure, &grid).unwrap();
    }

    #[test]
    fn test_visualization_config_validation() {
        // Valid config
        let valid_config = VisualizationConfig::default();
        valid_config.validate().unwrap();

        // Invalid target_fps
        let invalid_config = VisualizationConfig {
            target_fps: 0.0,
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());

        // Invalid max_texture_size
        let invalid_config = VisualizationConfig {
            max_texture_size: 0,
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_metrics_summary() {
        let mut tracker = MetricsTracker::new();
        tracker.update(16.0, 2.0);
        tracker.update_memory(1_048_576);

        let summary = tracker.summary();
        assert!(summary.contains("FPS"));
        assert!(summary.contains("Render"));
        assert!(summary.contains("Transfer"));
        assert!(summary.contains("GPU Memory"));
    }
}
