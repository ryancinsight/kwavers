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
//! - **Typed Initialization Errors**: rendering before `initialize_gpu` returns an error, never a CPU substitute
//!
//! ## Architecture
//! ```text
//! VisualizationEngine
//! ├── Config (configuration management)
//! ├── Metrics (performance tracking)
//! ├── Engine (core rendering pipeline)
//! ├── Transfer contract (provider-neutral field upload)
//! └── Fallback (CPU field statistics, ASCII slices, and text export)
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

// Rendering, transfer, and control modules
pub mod controls;
pub mod data_pipeline;
pub mod renderer;
pub mod stream;
// Provider-neutral transfer seam; the domain crate owns the contract, the
// Hephaestus-backed WGPU implementation lives in kwavers-gpu.
pub mod transfer_contract;

// Re-exports for convenience
pub use config::{ColorScheme, RenderQuality, VisualizationConfig};
pub use engine::{UnconfiguredVisualizationProvider, VisualizationEngine};
pub use metrics::{MetricsTracker, VisualizationMetrics};
pub use transfer_contract::VisualizationTransferProvider;

// Re-export field types
pub use kwavers_field::mapping::UnifiedFieldType;

// Rendering and transfer re-exports
pub use controls::InteractiveControls;
pub use data_pipeline::{DataPipeline, TransferMode, TransferOptions};
pub use renderer::Renderer3D;
pub use stream::{FrameMetadata, VizFrame, VizStream};

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::error::{KwaversError, SystemError};
    use kwavers_grid::Grid;
    use leto::{Array3, Array4};
    use std::sync::{Arc, Mutex};

    type Transfer = (UnifiedFieldType, Vec<f32>, TransferMode);
    type TransferLog = Arc<Mutex<Vec<Transfer>>>;

    /// Transfer provider whose log outlives the engine that consumes it.
    #[derive(Debug, Default)]
    struct RecordingProvider {
        transfers: TransferLog,
    }

    fn recorded(transfers: &Mutex<Vec<Transfer>>) -> Vec<Transfer> {
        transfers
            .lock()
            .expect("invariant: no test panics while holding the transfer log")
            .clone()
    }

    impl VisualizationTransferProvider for RecordingProvider {
        fn device_name(&self) -> &'static str {
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
            self.transfers
                .lock()
                .expect("invariant: no test panics while holding the transfer log")
                .push((field_type, samples.to_vec(), mode));
            Ok(())
        }

        fn memory_usage(&self) -> usize {
            recorded(&self.transfers)
                .iter()
                .map(|(_, samples, _)| std::mem::size_of_val(samples.as_slice()))
                .sum()
        }
    }

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

    const FIELD_SHAPE: (usize, usize, usize) = (2, 3, 4);
    const FIELD_TYPES: [UnifiedFieldType; 3] = [
        UnifiedFieldType::Pressure,
        UnifiedFieldType::Temperature,
        UnifiedFieldType::LightFluence,
    ];

    /// The row-major position of a sample in a `FIELD_SHAPE` volume, offset by
    /// 24 per field, so that every sample of the three fields is distinct.
    fn sample_code(field: usize, i: usize, j: usize, k: usize) -> u8 {
        u8::try_from(24 * field + 12 * i + 4 * j + k)
            .expect("invariant: three 2x3x4 fields hold 72 samples")
    }

    fn coded_fields() -> Array4<f64> {
        let (nx, ny, nz) = FIELD_SHAPE;
        Array4::from_shape_fn((nx, ny, nz, FIELD_TYPES.len()), |[i, j, k, field]| {
            f64::from(sample_code(field, i, j, k))
        })
    }

    fn coded_grid() -> Grid {
        let (nx, ny, nz) = FIELD_SHAPE;
        Grid::new(nx, ny, nz, 1e-3, 1e-3, 1e-3).expect("test grid is valid")
    }

    fn initialized_test_engine(
        config: VisualizationConfig,
    ) -> (VisualizationEngine<RecordingProvider>, TransferLog) {
        let provider = RecordingProvider::default();
        let transfers = Arc::clone(&provider.transfers);
        let mut engine = VisualizationEngine::create(config)
            .expect("valid visualization configuration")
            .set_transfer_provider(provider);
        engine
            .initialize_gpu()
            .expect("recording provider initializes the pipeline");
        (engine, transfers)
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
        let mut engine = create_configured_test_engine(config);
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
    fn test_render_multi_field_rejects_zero_fields() {
        let (mut engine, transfers) = initialized_test_engine(VisualizationConfig::default());
        let fields = Array4::zeros((2, 3, 4, 0));

        let result = engine.render_multi_field(&fields, &[], &coded_grid());

        assert!(matches!(
            result,
            Err(KwaversError::InvalidInput(message))
                if message == "multi-field rendering requires at least one field"
        ));
        assert_eq!(recorded(&transfers), Vec::new());
    }

    #[test]
    fn test_initialized_render_multi_field_transfers_every_field() {
        let (mut engine, transfers) = initialized_test_engine(VisualizationConfig::default());

        engine
            .render_multi_field(&coded_fields(), &FIELD_TYPES, &coded_grid())
            .expect("an initialized engine renders three fields");

        let (nx, ny, nz) = FIELD_SHAPE;
        let expected: Vec<Transfer> = FIELD_TYPES
            .iter()
            .enumerate()
            .map(|(field, &field_type)| {
                let mut samples = Vec::with_capacity(nx * ny * nz);
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            samples.push(f32::from(sample_code(field, i, j, k)));
                        }
                    }
                }
                (field_type, samples, TransferMode::Async)
            })
            .collect();
        assert_eq!(recorded(&transfers), expected);
    }

    #[test]
    fn test_render_multi_field_hands_later_fields_to_the_renderer() {
        // The renderer composites only with transparency, so its rejection
        // shows that it was handed more than the first field.
        let config = VisualizationConfig {
            enable_transparency: false,
            ..VisualizationConfig::default()
        };
        let (mut engine, _transfers) = initialized_test_engine(config);

        let result = engine.render_multi_field(&coded_fields(), &FIELD_TYPES, &coded_grid());

        assert!(matches!(
            result,
            Err(KwaversError::InvalidInput(message))
                if message == "multi-field rendering requires transparency for compositing"
        ));
    }

    #[test]
    fn test_render_multi_field_after_cleanup_requires_initialization() {
        let (mut engine, transfers) = initialized_test_engine(VisualizationConfig::default());
        assert_eq!(Arc::strong_count(&transfers), 2);
        engine.cleanup();
        // The pipeline owned the provider, and the provider the other handle
        // on the log.
        assert_eq!(Arc::strong_count(&transfers), 1);

        let result = engine.render_multi_field(&coded_fields(), &FIELD_TYPES, &coded_grid());

        assert!(matches!(
            result,
            Err(KwaversError::System(SystemError::FeatureNotAvailable { feature, .. }))
                if feature == "gpu-visualization"
        ));
        assert_eq!(recorded(&transfers), Vec::new());
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
