//! Visualization Configuration Module
//!
//! Provides configuration structures for visualization and rendering.

use crate::core::error::KwaversResult;

// Constants for visualization
pub const DEFAULT_TARGET_FPS: f64 = 60.0;
pub const LOW_TARGET_FPS: f64 = 30.0;
pub const DEFAULT_MAX_TEXTURE_SIZE: usize = 512;
pub const MEDIUM_GRID_SIZE: usize = 128;
pub const MILLISECONDS_PER_SECOND: f64 = 1000.0;

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
    /// Magma color scheme
    Magma,
}

/// Render quality settings
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderQuality {
    /// Low quality for real-time interaction (>60 FPS)
    Low,
    /// Medium quality for balanced performance (30-60 FPS)
    Medium,
    /// High quality for final visualization (<30 FPS)
    High,
    /// Draft quality for quick previews
    Draft,
    /// Production quality for general use
    Production,
    /// Publication quality for high-quality output
    Publication,
}

/// Visualization configuration
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Target frames per second
    pub target_fps: f64,
    /// Render quality setting
    pub quality: RenderQuality,
    /// Render quality setting (alias for backwards compatibility)
    pub render_quality: RenderQuality,
    /// Color mapping scheme
    pub color_scheme: ColorScheme,
    /// Enable transparency for multi-field rendering
    pub enable_transparency: bool,
    /// Maximum texture size for GPU uploads
    pub max_texture_size: usize,
    /// Enable performance profiling
    pub enable_profiling: bool,
    /// Ray marching samples for volume rendering
    pub ray_samples: usize,
    /// Enable GPU acceleration
    pub gpu_enabled: bool,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            target_fps: DEFAULT_TARGET_FPS,
            quality: RenderQuality::Medium,
            render_quality: RenderQuality::Medium,
            color_scheme: ColorScheme::Viridis,
            enable_transparency: true,
            max_texture_size: DEFAULT_MAX_TEXTURE_SIZE,
            enable_profiling: false,
            ray_samples: 128,
            gpu_enabled: true,
        }
    }
}

impl VisualizationConfig {
    /// Create a configuration optimized for performance
    pub fn performance() -> Self {
        Self {
            target_fps: DEFAULT_TARGET_FPS,
            quality: RenderQuality::Low,
            render_quality: RenderQuality::Low,
            color_scheme: ColorScheme::Grayscale,
            enable_transparency: false,
            max_texture_size: 256,
            enable_profiling: false,
            ray_samples: 64,
            gpu_enabled: true,
        }
    }

    /// Create a configuration optimized for quality
    pub fn quality() -> Self {
        Self {
            target_fps: LOW_TARGET_FPS,
            quality: RenderQuality::High,
            render_quality: RenderQuality::High,
            color_scheme: ColorScheme::Viridis,
            enable_transparency: true,
            max_texture_size: 1024,
            enable_profiling: false,
            ray_samples: 256,
            gpu_enabled: true,
        }
    }

    /// Create a configuration for debugging
    pub fn debug() -> Self {
        Self {
            target_fps: LOW_TARGET_FPS,
            quality: RenderQuality::Low,
            render_quality: RenderQuality::Low,
            color_scheme: ColorScheme::Turbo,
            enable_transparency: false,
            max_texture_size: DEFAULT_MAX_TEXTURE_SIZE,
            enable_profiling: true,
            ray_samples: 64,
            gpu_enabled: false,
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> KwaversResult<()> {
        if self.target_fps <= 0.0 {
            return Err(crate::core::error::KwaversError::Validation(
                crate::core::error::ValidationError::OutOfRange {
                    value: self.target_fps,
                    min: 0.0,
                    max: f64::INFINITY,
                }, /* field: target_fps */
            ));
        }

        if self.max_texture_size == 0 {
            return Err(crate::core::error::KwaversError::Validation(
                crate::core::error::ValidationError::OutOfRange {
                    value: self.max_texture_size as f64,
                    min: 1.0,
                    max: f64::INFINITY,
                }, /* field: max_texture_size */
            ));
        }

        Ok(())
    }
}
