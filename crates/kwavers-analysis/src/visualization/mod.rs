//! # Visualization Module
//!
//! Core visualization and rendering infrastructure for the acoustic simulation library.
//!
//! ## Features
//! - **Real-time Rendering**: GPU-accelerated visualization of acoustic fields
//! - **Multiple Backends**: Support for OpenGL, Vulkan, and WebGPU
//! - **Adaptive Quality**: Dynamic quality adjustment based on performance
//! - **Interactive Controls**: Real-time parameter adjustment and view manipulation
//! - **Data Export**: High-quality image and video export capabilities
//! - **Fallback Support**: CPU-based rendering when GPU is unavailable
//!
//! ## Architecture
//! ```text
//! VisualizationEngine
//! ├── Config (configuration management)
//! ├── Metrics (performance tracking)
//! ├── Engine (core rendering pipeline)
//! └── Fallback (CPU-based rendering)
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

// GPU-specific modules
#[cfg(feature = "gpu-visualization")]
pub mod controls;
#[cfg(feature = "gpu-visualization")]
pub mod data_pipeline;
#[cfg(feature = "gpu-visualization")]
pub mod renderer;
#[cfg(feature = "gpu-visualization")]
pub mod stream;

// Re-exports for convenience
pub use config::{ColorScheme, RenderQuality, VisualizationConfig};
pub use engine::VisualizationEngine;
pub use metrics::{MetricsTracker, VisualizationMetrics};

// Re-export field types
pub use kwavers_field::mapping::UnifiedFieldType;

// GPU-specific re-exports
#[cfg(feature = "gpu-visualization")]
pub use controls::InteractiveControls;
#[cfg(feature = "gpu-visualization")]
pub use data_pipeline::DataPipeline;
#[cfg(feature = "gpu-visualization")]
pub use renderer::Renderer3D;
#[cfg(feature = "gpu-visualization")]
pub use stream::{FrameMetadata, VizFrame, VizStream};

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_grid::Grid;
    use leto::{Array3, Array4};

    fn create_test_grid() -> Grid {
        Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).expect("Failed to create test grid")
    }

    fn create_test_field() -> Array3<f64> {
        Array3::zeros((32, 32, 32))
    }

    #[test]
    fn test_visualization_config_default() {
        let config = VisualizationConfig::default();
        assert_eq!(config.target_frame_rate, config::DEFAULT_TARGET_FRAME_RATE);
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
        assert_eq!(
            metrics.frame_rate,
            aequitas::systems::si::quantities::Frequency::from_base(0.0)
        );
        assert_eq!(metrics.gpu_memory_usage, 0);
        assert_eq!(
            metrics.render_time,
            aequitas::systems::si::quantities::Time::from_base(0.0)
        );
        assert_eq!(
            metrics.transfer_time,
            aequitas::systems::si::quantities::Time::from_base(0.0)
        );
        assert_eq!(metrics.rendered_primitives, 0);
    }

    #[test]
    fn test_metrics_tracker() {
        let mut tracker = MetricsTracker::new();

        // Update with some measurements
        tracker.update(
            aequitas::systems::si::quantities::Time::from_base(16.0e-3),
            aequitas::systems::si::quantities::Time::from_base(2.0e-3),
        );
        tracker.update(
            aequitas::systems::si::quantities::Time::from_base(15.0e-3),
            aequitas::systems::si::quantities::Time::from_base(3.0e-3),
        );
        tracker.update(
            aequitas::systems::si::quantities::Time::from_base(17.0e-3),
            aequitas::systems::si::quantities::Time::from_base(2.5e-3),
        );

        let metrics = tracker.current();
        assert!(metrics.frame_rate.into_base() > 0.0);
        assert!(metrics.render_time.into_base() > 0.0);
        assert!(metrics.transfer_time.into_base() > 0.0);
    }

    #[test]
    fn test_performance_targets() {
        let mut tracker = MetricsTracker::new();

        // Simulate good performance
        for _ in 0..10 {
            tracker.update(
                aequitas::systems::si::quantities::Time::from_base(10.0e-3),
                aequitas::systems::si::quantities::Time::from_base(2.0e-3),
            ); // 12 ms total = ~83 FPS
        }

        assert!(
            tracker.meets_target(aequitas::systems::si::quantities::Frequency::from_base(
                60.0
            ))
        );
        assert!(
            !tracker.meets_target(aequitas::systems::si::quantities::Frequency::from_base(
                100.0
            ))
        );
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

    #[test]
    fn test_render_field_without_gpu() {
        let config = VisualizationConfig::default();
        let mut engine = VisualizationEngine::create(config).unwrap();

        let grid = create_test_grid();
        let field = create_test_field();

        pollster::block_on(engine.render_field(&field, UnifiedFieldType::Pressure, &grid)).unwrap();
    }

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

        pollster::block_on(engine.render_multi_field(&fields, &field_types, &grid)).unwrap();
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

        // Invalid target frame rate
        let invalid_config = VisualizationConfig {
            target_frame_rate: aequitas::systems::si::quantities::Frequency::from_base(0.0),
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
        tracker.update(
            aequitas::systems::si::quantities::Time::from_base(16.0e-3),
            aequitas::systems::si::quantities::Time::from_base(2.0e-3),
        );
        tracker.update_memory(1_048_576);

        let summary = tracker.summary();
        assert!(summary.contains("FPS"));
        assert!(summary.contains("Render"));
        assert!(summary.contains("Transfer"));
        assert!(summary.contains("GPU Memory"));
    }
}
