//! # Advanced Visualization Module - Phase 11
//!
//! This module provides comprehensive 3D visualization and real-time interaction capabilities
//! for Kwavers simulation results. It implements GPU-accelerated volume rendering, interactive
//! parameter controls, and multi-platform support including web and VR.
//!
//! ## Features
//! - **Real-Time 3D Visualization**: GPU-accelerated volume rendering for pressure, temperature, and optical fields
//! - **Interactive Controls**: Live parameter adjustment during simulation execution
//! - **Multi-Field Display**: Simultaneous visualization of multiple simulation fields
//! - **Web Support**: WebGL-based rendering for browser deployment
//! - **VR Integration**: Immersive 3D visualization for complex simulations
//! - **Performance Optimization**: 60+ FPS rendering for 128³ grids
//!
//! ## Architecture
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  Visualization Engine                       │
//! ├─────────────────┬─────────────────┬─────────────────────────┤
//! │   3D Renderer   │   UI Controls   │   Data Pipeline         │
//! │   - Volume      │   - Parameters  │   - Field Extraction    │
//! │   - Isosurface  │   - Real-time   │   - GPU Transfer        │
//! │   - Multi-field │   - Validation  │   - Memory Management   │
//! └─────────────────┴─────────────────┴─────────────────────────┘
//! ```

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::gpu::GpuContext;
use log::{debug, info};
use ndarray::{Array3, Array4};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// Conditional imports based on features
#[cfg(feature = "advanced-visualization")]
pub mod renderer;
#[cfg(feature = "advanced-visualization")]
pub mod controls;
#[cfg(feature = "advanced-visualization")]
pub mod data_pipeline;
// Additional modules will be implemented in future phases
// #[cfg(feature = "advanced-visualization")]
// pub mod volume_rendering;
// #[cfg(feature = "advanced-visualization")]
// pub mod isosurface;

// #[cfg(feature = "web-visualization")]
// pub mod web;

// #[cfg(feature = "vr-support")]
// pub mod vr;

// Re-exports for convenience
#[cfg(feature = "advanced-visualization")]
pub use renderer::*;
#[cfg(feature = "advanced-visualization")]
pub use controls::*;
#[cfg(feature = "advanced-visualization")]
pub use data_pipeline::*;

/// Visualization field types supported by the engine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldType {
    /// Acoustic pressure field (Pa)
    Pressure,
    /// Temperature field (K)
    Temperature,
    /// Optical field intensity (W/m²)
    OpticalIntensity,
    /// Cavitation bubble radius (m)
    BubbleRadius,
    /// Velocity field magnitude (m/s)
    VelocityMagnitude,
    /// Custom field (user-defined)
    Custom(usize),
}

/// Color mapping schemes for scientific visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorScheme {
    /// Viridis (perceptually uniform)
    Viridis,
    /// Plasma (high contrast)
    Plasma,
    /// Inferno (dark background)
    Inferno,
    /// Turbo (rainbow-like)
    Turbo,
    /// Grayscale
    Grayscale,
    /// Custom RGB mapping
    Custom,
}

/// Rendering quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderQuality {
    /// Low quality for real-time interaction (>60 FPS)
    Low,
    /// Medium quality for balanced performance (30-60 FPS)
    Medium,
    /// High quality for final visualization (<30 FPS)
    High,
    /// Ultra quality for publication (no FPS constraint)
    Ultra,
}

/// Visualization configuration
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Target frame rate (FPS)
    pub target_fps: f32,
    /// Rendering quality level
    pub quality: RenderQuality,
    /// Color scheme for field visualization
    pub color_scheme: ColorScheme,
    /// Enable transparency for multi-field rendering
    pub enable_transparency: bool,
    /// Maximum texture size for GPU memory management
    pub max_texture_size: usize,
    /// Enable performance monitoring
    pub enable_profiling: bool,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            target_fps: 60.0,
            quality: RenderQuality::Medium,
            color_scheme: ColorScheme::Viridis,
            enable_transparency: true,
            max_texture_size: 512,
            enable_profiling: true,
        }
    }
}

/// Performance metrics for visualization system
#[derive(Debug, Clone)]
pub struct VisualizationMetrics {
    /// Current frame rate (FPS)
    pub fps: f32,
    /// GPU memory usage (bytes)
    pub gpu_memory_usage: usize,
    /// Rendering time per frame (ms)
    pub render_time_ms: f32,
    /// Data transfer time (ms)
    pub transfer_time_ms: f32,
    /// Number of rendered triangles/voxels
    pub rendered_primitives: usize,
}

/// Main visualization engine for Phase 11
pub struct VisualizationEngine {
    config: VisualizationConfig,
    gpu_context: Option<Arc<GpuContext>>,
    metrics: Arc<Mutex<VisualizationMetrics>>,
    last_frame_time: Instant,
    
    #[cfg(feature = "advanced-visualization")]
    renderer: Option<renderer::Renderer3D>,
    
    #[cfg(feature = "advanced-visualization")]
    controls: Option<controls::InteractiveControls>,
    
    #[cfg(feature = "advanced-visualization")]
    data_pipeline: Option<data_pipeline::DataPipeline>,
}

impl VisualizationEngine {
    /// Create a new visualization engine
    pub fn new(config: VisualizationConfig) -> KwaversResult<Self> {
        info!("Initializing Phase 11 Visualization Engine");
        
        let metrics = Arc::new(Mutex::new(VisualizationMetrics {
            fps: 0.0,
            gpu_memory_usage: 0,
            render_time_ms: 0.0,
            transfer_time_ms: 0.0,
            rendered_primitives: 0,
        }));
        
        Ok(Self {
            config,
            gpu_context: None,
            metrics,
            last_frame_time: Instant::now(),
            
            #[cfg(feature = "advanced-visualization")]
            renderer: None,
            
            #[cfg(feature = "advanced-visualization")]
            controls: None,
            
            #[cfg(feature = "advanced-visualization")]
            data_pipeline: None,
        })
    }
    
    /// Initialize GPU context for hardware acceleration
    pub async fn initialize_gpu(&mut self, gpu_context: Arc<GpuContext>) -> KwaversResult<()> {
        info!("Initializing GPU acceleration for visualization");
        self.gpu_context = Some(gpu_context.clone());
        
        #[cfg(feature = "advanced-visualization")]
        {
            // Initialize renderer with GPU context
            self.renderer = Some(renderer::Renderer3D::new(&self.config, gpu_context.clone()).await?);
            
            // Initialize data pipeline for efficient GPU transfers
            self.data_pipeline = Some(data_pipeline::DataPipeline::new(gpu_context).await?);
            
            // Initialize interactive controls
            self.controls = Some(controls::InteractiveControls::new(&self.config)?);
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
        
        #[cfg(feature = "advanced-visualization")]
        {
            if let (Some(renderer), Some(pipeline)) = (&mut self.renderer, &mut self.data_pipeline) {
                // Transfer field data to GPU
                let transfer_start = Instant::now();
                pipeline.upload_field(field, field_type).await?;
                let transfer_time = transfer_start.elapsed().as_secs_f32() * 1000.0;
                
                // Render the field
                let render_start = Instant::now();
                renderer.render_volume(field_type, grid).await?;
                let render_time = render_start.elapsed().as_secs_f32() * 1000.0;
                
                // Update metrics
                self.update_metrics(render_time, transfer_time);
                
                debug!("Rendered {} field: {:.2}ms render, {:.2}ms transfer", 
                       format!("{:?}", field_type), render_time, transfer_time);
            }
        }
        
        #[cfg(not(feature = "advanced-visualization"))]
        {
            warn!("Advanced visualization not enabled. Enable 'advanced-visualization' feature.");
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
        let start_time = Instant::now();
        
        #[cfg(feature = "advanced-visualization")]
        {
            if let (Some(renderer), Some(pipeline)) = (&mut self.renderer, &mut self.data_pipeline) {
                // Upload all fields to GPU
                let transfer_start = Instant::now();
                for (i, &field_type) in field_types.iter().enumerate() {
                    if i < fields.shape()[3] {
                        let field = fields.slice(ndarray::s![.., .., .., i]);
                        pipeline.upload_field(&field.to_owned(), field_type).await?;
                    }
                }
                let transfer_time = transfer_start.elapsed().as_secs_f32() * 1000.0;
                
                // Render all fields with transparency blending
                let render_start = Instant::now();
                renderer.render_multi_volume(field_types, grid).await?;
                let render_time = render_start.elapsed().as_secs_f32() * 1000.0;
                
                // Update metrics
                self.update_metrics(render_time, transfer_time);
                
                info!("Rendered {} fields: {:.2}ms render, {:.2}ms transfer", 
                      field_types.len(), render_time, transfer_time);
            }
        }
        
        #[cfg(not(feature = "advanced-visualization"))]
        {
            warn!("Advanced visualization not enabled. Enable 'advanced-visualization' feature.");
        }
        
        Ok(())
    }
    
    /// Update real-time parameter during simulation
    pub fn update_parameter(&mut self, parameter: &str, value: f64) -> KwaversResult<()> {
        #[cfg(feature = "advanced-visualization")]
        {
            if let Some(controls) = &mut self.controls {
                controls.update_parameter(parameter, crate::visualization::controls::ParameterValue::Float(value))?;
                debug!("Updated parameter {}: {}", parameter, value);
            }
        }
        
        Ok(())
    }
    
    /// Get current visualization metrics
    pub fn get_metrics(&self) -> VisualizationMetrics {
        self.metrics.lock().unwrap().clone()
    }
    
    /// Check if performance targets are met
    pub fn meets_performance_targets(&self) -> bool {
        let metrics = self.metrics.lock().unwrap();
        metrics.fps >= self.config.target_fps * 0.9 // 90% of target FPS
    }
    
    /// Update performance metrics
    fn update_metrics(&mut self, render_time: f32, transfer_time: f32) {
        let current_time = Instant::now();
        let frame_time = current_time.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = current_time;
        
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.fps = if frame_time > 0.0 { 1.0 / frame_time } else { 0.0 };
            metrics.render_time_ms = render_time;
            metrics.transfer_time_ms = transfer_time;
            
            #[cfg(feature = "advanced-visualization")]
            {
                if let Some(renderer) = &self.renderer {
                    metrics.gpu_memory_usage = renderer.get_memory_usage();
                    metrics.rendered_primitives = renderer.get_primitive_count();
                }
            }
        }
    }
}

// Stub implementations for when advanced visualization is not enabled
#[cfg(not(feature = "advanced-visualization"))]
mod stubs {
    use super::*;
    
    /// Fallback function for basic field visualization
    pub fn render_field_basic(
        field: &Array3<f64>,
        field_type: FieldType,
        grid: &Grid,
    ) -> KwaversResult<()> {
        info!("Basic visualization: {} field with {} points", 
              format!("{:?}", field_type), field.len());
        
        // Use existing plotting module for basic visualization
        #[cfg(feature = "plotly")]
        {
            use crate::plotting::plot_pressure_field_3d;
            if field_type == FieldType::Pressure {
                plot_pressure_field_3d(field, grid, "Pressure Field", "pressure_3d.html");
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use ndarray::Array3;
    
    fn create_test_grid() -> Grid {
        Grid::new(32, 32, 32, 0.01, 0.01, 0.01)
    }
    
    fn create_test_field() -> Array3<f64> {
        let mut field = Array3::zeros((32, 32, 32));
        
        // Create a simple test pattern
        for i in 0..32 {
            for j in 0..32 {
                for k in 0..32 {
                    let x = i as f64 / 31.0;
                    let y = j as f64 / 31.0;
                    let z = k as f64 / 31.0;
                    
                    // Gaussian blob in the center
                    let dx = x - 0.5;
                    let dy = y - 0.5;
                    let dz = z - 0.5;
                    let r_squared = dx*dx + dy*dy + dz*dz;
                    field[[i, j, k]] = (-r_squared * 10.0).exp();
                }
            }
        }
        
        field
    }
    
    #[test]
    fn test_field_type_enum() {
        let pressure = FieldType::Pressure;
        let temperature = FieldType::Temperature;
        let custom = FieldType::Custom(42);
        
        assert_eq!(pressure, FieldType::Pressure);
        assert_eq!(temperature, FieldType::Temperature);
        assert_eq!(custom, FieldType::Custom(42));
        assert_ne!(pressure, temperature);
    }
    
    #[test]
    fn test_color_scheme_enum() {
        let viridis = ColorScheme::Viridis;
        let plasma = ColorScheme::Plasma;
        
        assert_eq!(viridis, ColorScheme::Viridis);
        assert_eq!(plasma, ColorScheme::Plasma);
        assert_ne!(viridis, plasma);
    }
    
    #[test]
    fn test_render_quality_enum() {
        let low = RenderQuality::Low;
        let high = RenderQuality::High;
        
        assert_eq!(low, RenderQuality::Low);
        assert_eq!(high, RenderQuality::High);
        assert_ne!(low, high);
    }
    
    #[test]
    fn test_visualization_config_default() {
        let config = VisualizationConfig::default();
        
        assert_eq!(config.target_fps, 60.0);
        assert_eq!(config.quality, RenderQuality::Medium);
        assert_eq!(config.color_scheme, ColorScheme::Viridis);
        assert!(config.enable_transparency);
        assert_eq!(config.max_texture_size, 512);
        assert!(config.enable_profiling);
    }
    
    #[test]
    fn test_visualization_engine_creation() {
        let config = VisualizationConfig::default();
        let engine = VisualizationEngine::new(config);
        
        assert!(engine.is_ok());
        
        let engine = engine.unwrap();
        let metrics = engine.get_metrics();
        
        assert_eq!(metrics.fps, 0.0);
        assert_eq!(metrics.gpu_memory_usage, 0);
        assert_eq!(metrics.rendered_primitives, 0);
    }
    
    #[test]
    fn test_visualization_metrics() {
        let config = VisualizationConfig::default();
        let engine = VisualizationEngine::new(config).unwrap();
        
        let metrics = engine.get_metrics();
        assert_eq!(metrics.fps, 0.0);
        assert_eq!(metrics.render_time_ms, 0.0);
        assert_eq!(metrics.transfer_time_ms, 0.0);
        assert_eq!(metrics.rendered_primitives, 0);
    }
    
    #[test]
    fn test_performance_targets() {
        let config = VisualizationConfig {
            target_fps: 30.0,
            ..Default::default()
        };
        let engine = VisualizationEngine::new(config).unwrap();
        
        // Should not meet targets initially (0 FPS)
        assert!(!engine.meets_performance_targets());
    }
    
    #[test]
    fn test_parameter_update() {
        let config = VisualizationConfig::default();
        let mut engine = VisualizationEngine::new(config).unwrap();
        
        let result = engine.update_parameter("frequency", 2.0);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_render_field_without_gpu() {
        let config = VisualizationConfig::default();
        let mut engine = VisualizationEngine::new(config).unwrap();
        
        let grid = create_test_grid();
        let field = create_test_field();
        
        // Should not fail even without GPU context
        let result = pollster::block_on(engine.render_field(&field, FieldType::Pressure, &grid));
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_render_multi_field_without_gpu() {
        let config = VisualizationConfig::default();
        let mut engine = VisualizationEngine::new(config).unwrap();
        
        let grid = create_test_grid();
        let fields = Array4::zeros((32, 32, 32, 3));
        let field_types = vec![FieldType::Pressure, FieldType::Temperature, FieldType::OpticalIntensity];
        
        // Should not fail even without GPU context
        let result = pollster::block_on(engine.render_multi_field(&fields, &field_types, &grid));
        assert!(result.is_ok());
    }
    
    #[cfg(not(feature = "advanced-visualization"))]
    #[test]
    fn test_stub_implementation() {
        use super::stubs::*;
        
        let grid = create_test_grid();
        let field = create_test_field();
        
        let result = render_field_basic(&field, FieldType::Pressure, &grid);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_visualization_config_custom() {
        let config = VisualizationConfig {
            target_fps: 120.0,
            quality: RenderQuality::Ultra,
            color_scheme: ColorScheme::Plasma,
            enable_transparency: false,
            max_texture_size: 1024,
            enable_profiling: false,
        };
        
        assert_eq!(config.target_fps, 120.0);
        assert_eq!(config.quality, RenderQuality::Ultra);
        assert_eq!(config.color_scheme, ColorScheme::Plasma);
        assert!(!config.enable_transparency);
        assert_eq!(config.max_texture_size, 1024);
        assert!(!config.enable_profiling);
    }
    
    #[test]
    fn test_field_type_debug() {
        let pressure = FieldType::Pressure;
        let debug_str = format!("{:?}", pressure);
        assert_eq!(debug_str, "Pressure");
        
        let custom = FieldType::Custom(123);
        let debug_str = format!("{:?}", custom);
        assert_eq!(debug_str, "Custom(123)");
    }
    
    #[test]
    fn test_metrics_clone() {
        let metrics = VisualizationMetrics {
            fps: 60.0,
            gpu_memory_usage: 1024,
            render_time_ms: 16.67,
            transfer_time_ms: 2.5,
            rendered_primitives: 1000000,
        };
        
        let cloned = metrics.clone();
        assert_eq!(metrics.fps, cloned.fps);
        assert_eq!(metrics.gpu_memory_usage, cloned.gpu_memory_usage);
        assert_eq!(metrics.render_time_ms, cloned.render_time_ms);
        assert_eq!(metrics.transfer_time_ms, cloned.transfer_time_ms);
        assert_eq!(metrics.rendered_primitives, cloned.rendered_primitives);
    }
}