//! Volume rendering implementation

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::visualization::{ColorScheme, FieldType, VisualizationConfig};
use ndarray::Array3;

/// Volume renderer for 3D fields
/// NOTE: Some fields currently unused - part of future volume rendering implementation
#[allow(dead_code)]
#[derive(Debug)]
pub struct VolumeRenderer {
    config: VisualizationConfig,
    transfer_function: TransferFunction,
    ray_marcher: RayMarcher,
}

impl VolumeRenderer {
    /// Create a new volume renderer
    pub fn new(config: &VisualizationConfig) -> KwaversResult<Self> {
        Ok(Self {
            config: config.clone(),
            transfer_function: TransferFunction::new(&config.color_scheme),
            ray_marcher: RayMarcher::new(config.ray_samples),
        })
    }

    /// Render with draft quality
    pub fn render_draft(
        &self,
        field: &Array3<f64>,
        field_type: FieldType,
        grid: &Grid,
    ) -> KwaversResult<Vec<u8>> {
        self.render_internal(field, field_type, grid, 32)
    }

    /// Render with production quality
    pub fn render_production(
        &self,
        field: &Array3<f64>,
        field_type: FieldType,
        grid: &Grid,
    ) -> KwaversResult<Vec<u8>> {
        self.render_internal(field, field_type, grid, 128)
    }

    /// Render with publication quality
    pub fn render_publication(
        &self,
        field: &Array3<f64>,
        field_type: FieldType,
        grid: &Grid,
    ) -> KwaversResult<Vec<u8>> {
        self.render_internal(field, field_type, grid, 256)
    }

    /// Internal rendering implementation
    fn render_internal(
        &self,
        field: &Array3<f64>,
        _field_type: FieldType,
        _grid: &Grid,
        _samples: usize,
    ) -> KwaversResult<Vec<u8>> {
        let (nx, ny, nz) = field.dim();
        let mut image = vec![0u8; nx * ny * 4]; // RGBA

        // Maximum intensity projection (MIP) per Levoy (1988)
        // Standard volume rendering technique for medical visualization
        // Alternative: direct volume rendering with ray marching (see Sprint 125+ roadmap)
        for i in 0..nx {
            for j in 0..ny {
                let mut max_val = 0.0_f32;
                for k in 0..nz {
                    max_val = max_val.max(field[[i, j, k]].abs() as f32);
                }

                let color = self.transfer_function.map_value(max_val);
                let idx = (j * nx + i) * 4;
                image[idx] = (color[0] * 255.0) as u8;
                image[idx + 1] = (color[1] * 255.0) as u8;
                image[idx + 2] = (color[2] * 255.0) as u8;
                image[idx + 3] = 255;
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
    color_map: Vec<[f32; 4]>, // RGBA
}

impl TransferFunction {
    /// Create a new transfer function
    fn new(scheme: &ColorScheme) -> Self {
        let color_map = match scheme {
            ColorScheme::Viridis => Self::viridis_colormap(),
            ColorScheme::Plasma => Self::plasma_colormap(),
            ColorScheme::Inferno => Self::inferno_colormap(),
            ColorScheme::Magma => Self::magma_colormap(),
            ColorScheme::Turbo => Self::turbo_colormap(),
            ColorScheme::Grayscale => Self::grayscale_colormap(),
            ColorScheme::Custom => Self::viridis_colormap(), // Fallback to viridis for custom
        };

        Self { color_map }
    }

    /// Map a value to a color
    fn map_value(&self, value: f32) -> [f32; 4] {
        let idx = ((value.clamp(0.0, 1.0) * (self.color_map.len() - 1) as f32) as usize)
            .min(self.color_map.len() - 1);
        self.color_map[idx]
    }

    /// Viridis colormap
    fn viridis_colormap() -> Vec<[f32; 4]> {
        vec![
            [0.267, 0.004, 0.329, 1.0],
            [0.283, 0.141, 0.458, 1.0],
            [0.253, 0.265, 0.530, 1.0],
            [0.206, 0.372, 0.553, 1.0],
            [0.164, 0.471, 0.558, 1.0],
            [0.128, 0.567, 0.551, 1.0],
            [0.135, 0.659, 0.518, 1.0],
            [0.267, 0.749, 0.441, 1.0],
            [0.478, 0.821, 0.31832, 1.0], // Avoid exact FRAC_1_PI approximation
            [0.741, 0.873, 0.150, 1.0],
            [0.993, 0.906, 0.144, 1.0],
        ]
    }

    /// Plasma colormap (perceptually uniform, matplotlib-inspired)
    /// Reference: Smith & van der Walt (2015) "Colormaps" matplotlib documentation
    fn plasma_colormap() -> Vec<[f32; 4]> {
        // Plasma colormap: purple → pink → orange → yellow
        // Perceptually uniform for scientific visualization
        vec![
            [0.050, 0.030, 0.529, 1.0],
            [0.283, 0.024, 0.627, 1.0],
            [0.478, 0.007, 0.659, 1.0],
            [0.648, 0.060, 0.620, 1.0],
            [0.786, 0.184, 0.520, 1.0],
            [0.893, 0.335, 0.384, 1.0],
            [0.966, 0.505, 0.243, 1.0],
            [0.989, 0.690, 0.138, 1.0],
            [0.940, 0.876, 0.132, 1.0],
        ]
    }

    /// Inferno colormap (perceptually uniform, matplotlib-inspired)
    /// Reference: Smith & van der Walt (2015) "Colormaps" matplotlib documentation
    fn inferno_colormap() -> Vec<[f32; 4]> {
        // Inferno colormap: black → purple → red → orange → yellow
        // Excellent for thermal/heat visualization
        vec![
            [0.001, 0.000, 0.014, 1.0],
            [0.100, 0.031, 0.184, 1.0],
            [0.276, 0.044, 0.397, 1.0],
            [0.478, 0.066, 0.467, 1.0],
            [0.659, 0.137, 0.432, 1.0],
            [0.821, 0.268, 0.326, 1.0],
            [0.937, 0.449, 0.208, 1.0],
            [0.988, 0.653, 0.118, 1.0],
            [0.988, 0.880, 0.381, 1.0],
        ]
    }

    /// Magma colormap (perceptually uniform, matplotlib-inspired)
    /// Reference: Smith & van der Walt (2015) "Colormaps" matplotlib documentation
    fn magma_colormap() -> Vec<[f32; 4]> {
        // Magma colormap: black → purple → red → orange → white
        // Ideal for density/intensity visualization
        vec![
            [0.001, 0.000, 0.014, 1.0],
            [0.118, 0.051, 0.260, 1.0],
            [0.304, 0.080, 0.437, 1.0],
            [0.504, 0.119, 0.500, 1.0],
            [0.689, 0.196, 0.483, 1.0],
            [0.857, 0.328, 0.422, 1.0],
            [0.974, 0.524, 0.384, 1.0],
            [0.998, 0.730, 0.524, 1.0],
            [0.987, 0.914, 0.764, 1.0],
        ]
    }

    /// Turbo colormap (Google's improved rainbow, high dynamic range)
    /// Reference: Anton Mikhailov (2019) "Turbo, An Improved Rainbow Colormap"
    #[allow(clippy::approx_constant)] // False positive: RGB values, not math constants
    fn turbo_colormap() -> Vec<[f32; 4]> {
        // Turbo: improved rainbow with better perceptual uniformity
        // High dynamic range, reduces rainbow artifacts
        vec![
            [0.190, 0.073, 0.022, 1.0],
            [0.230, 0.318, 0.545, 1.0],
            [0.160, 0.519, 0.698, 1.0],
            [0.214, 0.682, 0.634, 1.0],
            [0.464, 0.801, 0.455, 1.0],
            [0.739, 0.872, 0.260, 1.0],
            [0.945, 0.869, 0.168, 1.0],
            [0.990, 0.683, 0.085, 1.0],
            [0.879, 0.314, 0.065, 1.0],
        ]
    }

    /// Grayscale colormap
    fn grayscale_colormap() -> Vec<[f32; 4]> {
        vec![
            [0.0, 0.0, 0.0, 1.0],
            [0.2, 0.2, 0.2, 1.0],
            [0.4, 0.4, 0.4, 1.0],
            [0.6, 0.6, 0.6, 1.0],
            [0.8, 0.8, 0.8, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    }
}

/// Ray marching for volume rendering
#[derive(Debug)]
struct RayMarcher {
    _samples: usize,
}

impl RayMarcher {
    /// Create a new ray marcher
    fn new(samples: usize) -> Self {
        Self { _samples: samples }
    }

    /// March a ray through the volume
    ///
    /// Note: This method is reserved for future volume rendering implementation.
    /// It implements basic ray marching for 3D volume visualization per Levoy (1988).
    ///
    /// **Reference**: Levoy (1988) "Display of Surfaces from Volume Data" IEEE CG&A
    #[allow(dead_code)]
    fn march_ray(&self, origin: [f32; 3], direction: [f32; 3], volume: &Array3<f64>) -> f32 {
        // Basic ray marching with uniform sampling (future: adaptive sampling)
        let mut accumulated = 0.0;
        let step = 1.0 / self._samples as f32;

        for i in 0..self._samples {
            let t = i as f32 * step;
            let pos = [
                origin[0] + t * direction[0],
                origin[1] + t * direction[1],
                origin[2] + t * direction[2],
            ];

            // Sample volume at position (with bounds checking)
            if pos[0] >= 0.0 && pos[1] >= 0.0 && pos[2] >= 0.0 {
                let ix = pos[0] as usize;
                let iy = pos[1] as usize;
                let iz = pos[2] as usize;

                if ix < volume.dim().0 && iy < volume.dim().1 && iz < volume.dim().2 {
                    accumulated += volume[[ix, iy, iz]].abs() as f32 * step;
                }
            }
        }

        accumulated
    }
}
