//! Volume rendering implementation

use crate::visualization::{ColorScheme, VisualizationConfig};
use iris::color::{ColorMap, NamedColorMap, Normalized};
use kwavers_core::error::KwaversResult;
use kwavers_field::UnifiedFieldType;
use kwavers_grid::Grid;
use leto::Array3;

/// Volume renderer for 3D fields
#[derive(Debug)]
pub struct VolumeRenderer {
    config: VisualizationConfig,
    transfer_function: TransferFunction,
}

impl VolumeRenderer {
    /// Create a new volume renderer
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(config: &VisualizationConfig) -> KwaversResult<Self> {
        Ok(Self {
            config: config.clone(),
            transfer_function: TransferFunction::new(&config.color_scheme),
        })
    }

    /// Render with draft quality
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn render_draft(
        &self,
        field: &Array3<f64>,
        field_type: UnifiedFieldType,
        grid: &Grid,
    ) -> KwaversResult<Vec<u8>> {
        self.render_internal(field, field_type, grid, 32)
    }

    /// Render with production quality
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn render_production(
        &self,
        field: &Array3<f64>,
        field_type: UnifiedFieldType,
        grid: &Grid,
    ) -> KwaversResult<Vec<u8>> {
        self.render_internal(field, field_type, grid, 128)
    }

    /// Render with publication quality
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn render_publication(
        &self,
        field: &Array3<f64>,
        field_type: UnifiedFieldType,
        grid: &Grid,
    ) -> KwaversResult<Vec<u8>> {
        self.render_internal(field, field_type, grid, 256)
    }

    /// Internal rendering implementation
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn render_internal(
        &self,
        field: &Array3<f64>,
        _field_type: UnifiedFieldType,
        _grid: &Grid,
        samples: usize,
    ) -> KwaversResult<Vec<u8>> {
        let [nx, ny, nz] = field.shape();
        let mut image = vec![0u8; nx * ny * 4]; // RGBA

        let global_max = field
            .iter()
            .fold(0.0_f32, |acc, &v| acc.max(v.abs() as f32));

        let step = nz.checked_div(samples).map_or(1, |q| q.max(1));

        for i in 0..nx {
            for j in 0..ny {
                let mut max_val = 0.0_f32;
                for k in (0..nz).step_by(step) {
                    max_val = max_val.max(field[[i, j, k]].abs() as f32);
                }

                let normalized = if global_max > 0.0 {
                    max_val / global_max
                } else {
                    0.0
                };
                let color = self.transfer_function.map_value(normalized);
                let idx = (j * nx + i) * 4;
                image[idx] = color[0];
                image[idx + 1] = color[1];
                image[idx + 2] = color[2];
                image[idx + 3] = if self.config.enable_transparency {
                    (normalized.clamp(0.0, 1.0) * 255.0) as u8
                } else {
                    255
                };
            }
        }

        Ok(image)
    }

    /// Get memory usage
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

/// Transfer function for mapping values to colors
#[derive(Debug)]
struct TransferFunction {
    color_map: NamedColorMap,
}

impl TransferFunction {
    /// Create a new transfer function
    fn new(scheme: &ColorScheme) -> Self {
        let color_map = match scheme {
            ColorScheme::Viridis => NamedColorMap::Viridis,
            ColorScheme::Plasma => NamedColorMap::Plasma,
            ColorScheme::Inferno => NamedColorMap::Inferno,
            ColorScheme::Magma => NamedColorMap::Magma,
            ColorScheme::Turbo => NamedColorMap::Turbo,
            ColorScheme::Grayscale => NamedColorMap::Grayscale,
            // Preserve prior fallback behavior for custom scheme.
            ColorScheme::Custom => NamedColorMap::Viridis,
        };

        Self { color_map }
    }

    /// Map a value to a color
    fn map_value(&self, value: f32) -> [u8; 4] {
        let quantized = (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        self.color_map
            .sample(Normalized::from_u8(quantized))
            .to_rgba8()
    }
}
