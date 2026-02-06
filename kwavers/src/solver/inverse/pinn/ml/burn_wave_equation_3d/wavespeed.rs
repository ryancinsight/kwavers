//! Wave speed function wrapper with Burn Module trait support
//!
//! This module provides a wrapper for wave speed functions c(x, y, z) that integrates
//! with Burn's Module system. It supports both CPU-based closures and device-resident
//! grids for efficient heterogeneous media modeling.
//!
//! ## Wave Speed Representations
//!
//! - **CPU Closure**: Function `c(x, y, z) → f32` evaluated on-demand
//! - **Device Grid**: Pre-computed tensor of wave speeds for fast lookup
//!
//! ## Module Integration
//!
//! Implements `Module`, `AutodiffModule`, and `ModuleDisplay` traits to enable:
//! - Device migration (CPU ↔ GPU)
//! - Automatic differentiation compatibility
//! - Module hierarchy inspection

use burn::module::{AutodiffModule, Devices, Module, ModuleMapper, ModuleVisitor};
use burn::tensor::{backend::AutodiffBackend, backend::Backend, Tensor};
use std::sync::Arc;

use crate::core::error::{KwaversError, KwaversResult, SystemError};

use super::geometry::Geometry3D;

#[derive(Debug, Clone)]
pub struct WaveSpeedGrid3D<B: Backend> {
    grid: Tensor<B, 3>,
    data_cpu: Arc<Vec<f32>>,
    dims: [usize; 3],
    bbox: [f32; 6],
}

impl<B: Backend> WaveSpeedGrid3D<B> {
    pub fn try_new(grid: Tensor<B, 3>, bbox: [f32; 6]) -> KwaversResult<Self> {
        let shape = grid.shape();
        let dims = match shape.dims.as_slice() {
            [nx, ny, nz] => [*nx, *ny, *nz],
            _ => {
                return Err(KwaversError::System(SystemError::InvalidConfiguration {
                    parameter: "wave_speed_grid".to_string(),
                    reason: format!(
                        "Expected wave speed grid with 3 dimensions, got {:?}",
                        shape.dims
                    ),
                }))
            }
        };
        let [nx, ny, nz] = dims;
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(KwaversError::System(SystemError::InvalidConfiguration {
                parameter: "wave_speed_grid".to_string(),
                reason: format!("Grid dimensions must be non-zero, got {dims:?}"),
            }));
        }

        let [x_min, x_max, y_min, y_max, z_min, z_max] = bbox;
        let bbox_ok = [x_min, x_max, y_min, y_max, z_min, z_max]
            .iter()
            .all(|v| v.is_finite())
            && x_min < x_max
            && y_min < y_max
            && z_min < z_max;
        if !bbox_ok {
            return Err(KwaversError::System(SystemError::InvalidConfiguration {
                parameter: "wave_speed_grid_bbox".to_string(),
                reason: format!("Invalid bounding box: {bbox:?}"),
            }));
        }

        let data = grid.clone().to_data();
        let slice = data.as_slice::<f32>().map_err(|e| {
            KwaversError::System(SystemError::InvalidConfiguration {
                parameter: "wave_speed_grid".to_string(),
                reason: format!("Expected f32 tensor data for wave speed grid: {e:?}"),
            })
        })?;

        let expected_len = nx
            .checked_mul(ny)
            .and_then(|v| v.checked_mul(nz))
            .ok_or_else(|| {
                KwaversError::System(SystemError::InvalidConfiguration {
                    parameter: "wave_speed_grid".to_string(),
                    reason: format!("Grid size overflows usize: {dims:?}"),
                })
            })?;
        if slice.len() != expected_len {
            return Err(KwaversError::System(SystemError::InvalidConfiguration {
                parameter: "wave_speed_grid".to_string(),
                reason: format!(
                    "Wave speed grid data length mismatch: expected {expected_len}, got {}",
                    slice.len()
                ),
            }));
        }

        if let Some((idx, &value)) = slice
            .iter()
            .enumerate()
            .find(|(_i, &v)| !v.is_finite() || v <= 0.0)
        {
            return Err(KwaversError::System(SystemError::InvalidConfiguration {
                parameter: "wave_speed_grid".to_string(),
                reason: format!(
                    "Wave speed grid contains invalid value at linear index {idx}: {value} (expected finite and > 0)"
                ),
            }));
        }

        let data_cpu = Arc::new(slice.to_vec());

        Ok(Self {
            grid,
            data_cpu,
            dims,
            bbox,
        })
    }

    pub fn grid(&self) -> &Tensor<B, 3> {
        &self.grid
    }

    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    pub fn bbox(&self) -> [f32; 6] {
        self.bbox
    }

    pub fn sample(&self, x: f32, y: f32, z: f32) -> f32 {
        let [x_min, x_max, y_min, y_max, z_min, z_max] = self.bbox;
        let [nx, ny, nz] = self.dims;
        let fx = Self::normalized_index(x, x_min, x_max, nx);
        let fy = Self::normalized_index(y, y_min, y_max, ny);
        let fz = Self::normalized_index(z, z_min, z_max, nz);

        let (x0, x1, wx) = Self::base_and_weight(fx, nx);
        let (y0, y1, wy) = Self::base_and_weight(fy, ny);
        let (z0, z1, wz) = Self::base_and_weight(fz, nz);

        let c000 = self.at(x0, y0, z0);
        let c100 = self.at(x1, y0, z0);
        let c010 = self.at(x0, y1, z0);
        let c110 = self.at(x1, y1, z0);
        let c001 = self.at(x0, y0, z1);
        let c101 = self.at(x1, y0, z1);
        let c011 = self.at(x0, y1, z1);
        let c111 = self.at(x1, y1, z1);

        let c00 = c000 + wx * (c100 - c000);
        let c10 = c010 + wx * (c110 - c010);
        let c01 = c001 + wx * (c101 - c001);
        let c11 = c011 + wx * (c111 - c011);

        let c0 = c00 + wy * (c10 - c00);
        let c1 = c01 + wy * (c11 - c01);

        c0 + wz * (c1 - c0)
    }

    fn at(&self, x: usize, y: usize, z: usize) -> f32 {
        let [_nx, ny, nz] = self.dims;
        let idx = (x * ny + y) * nz + z;
        self.data_cpu[idx]
    }

    fn normalized_index(v: f32, v_min: f32, v_max: f32, n: usize) -> f32 {
        if n <= 1 {
            return 0.0;
        }
        let denom = v_max - v_min;
        if denom == 0.0 {
            return 0.0;
        }
        let t = ((v - v_min) / denom).clamp(0.0, 1.0);
        t * ((n - 1) as f32)
    }

    fn base_and_weight(f: f32, n: usize) -> (usize, usize, f32) {
        if n <= 1 {
            return (0, 0, 0.0);
        }
        let f0 = f.floor();
        let i0 = (f0 as isize).clamp(0, (n - 1) as isize) as usize;
        let i1 = (i0 + 1).min(n - 1);
        let w = (f - f0).clamp(0.0, 1.0);
        (i0, i1, w)
    }
}

/// Wave speed function wrapper supporting CPU closures and device grids
///
/// This struct wraps a wave speed function c(x, y, z) and integrates it with
/// Burn's Module system for device management and autodiff compatibility.
///
/// # Type Parameters
///
/// * `B` - Backend type (e.g., NdArray, WGPU)
///
/// # Fields
///
/// * `func` - CPU closure for wave speed evaluation
/// * `grid` - Optional device-resident tensor for fast lookup
#[derive(Clone)]
pub struct WaveSpeedFn3D<B: Backend> {
    repr: WaveSpeedRepr3D<B>,
}

#[derive(Clone)]
enum WaveSpeedRepr3D<B: Backend> {
    Cpu(Arc<dyn Fn(f32, f32, f32) -> f32 + Send + Sync>),
    Grid(WaveSpeedGrid3D<B>),
}

impl<B: Backend> WaveSpeedFn3D<B> {
    /// Create a new wave speed function from a CPU closure
    ///
    /// # Arguments
    ///
    /// * `func` - Function returning wave speed at coordinates (x, y, z)
    ///
    /// # Returns
    ///
    /// A new `WaveSpeedFn3D` instance wrapping the closure
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::sync::Arc;
    /// use kwavers::solver::inverse::pinn::ml::burn_wave_equation_3d::WaveSpeedFn3D;
    ///
    /// // Constant wave speed
    /// let wave_speed = WaveSpeedFn3D::new(Arc::new(|_x, _y, _z| 1500.0));
    ///
    /// // Layered medium
    /// let layered = WaveSpeedFn3D::new(Arc::new(|_x, _y, z| {
    ///     if z < 0.5 { 1500.0 } else { 3000.0 }
    /// }));
    /// ```
    pub fn new(func: Arc<dyn Fn(f32, f32, f32) -> f32 + Send + Sync>) -> Self {
        Self {
            repr: WaveSpeedRepr3D::Cpu(func),
        }
    }

    pub fn from_grid_with_bbox(grid: Tensor<B, 3>, bbox: [f32; 6]) -> KwaversResult<Self> {
        Ok(Self {
            repr: WaveSpeedRepr3D::Grid(WaveSpeedGrid3D::try_new(grid, bbox)?),
        })
    }

    pub fn from_grid_with_geometry(
        grid: Tensor<B, 3>,
        geometry: &Geometry3D,
    ) -> KwaversResult<Self> {
        let (x_min, x_max, y_min, y_max, z_min, z_max) = geometry.bounding_box();
        Self::from_grid_with_bbox(
            grid,
            [
                x_min as f32,
                x_max as f32,
                y_min as f32,
                y_max as f32,
                z_min as f32,
                z_max as f32,
            ],
        )
    }

    /// Evaluate wave speed at coordinates (x, y, z)
    ///
    /// # Arguments
    ///
    /// * `x` - X-coordinate (meters)
    /// * `y` - Y-coordinate (meters)
    /// * `z` - Z-coordinate (meters)
    ///
    /// # Returns
    ///
    /// Wave speed c(x, y, z) in m/s
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let speed = wave_speed_fn.get(0.5, 0.5, 0.5);
    /// assert_eq!(speed, 1500.0);
    /// ```
    pub fn get(&self, x: f32, y: f32, z: f32) -> f32 {
        match &self.repr {
            WaveSpeedRepr3D::Cpu(func) => func(x, y, z),
            WaveSpeedRepr3D::Grid(grid) => grid.sample(x, y, z),
        }
    }

    pub fn has_grid(&self) -> bool {
        matches!(self.repr, WaveSpeedRepr3D::Grid(_))
    }

    pub fn grid_tensor(&self) -> Option<&Tensor<B, 3>> {
        match &self.repr {
            WaveSpeedRepr3D::Cpu(_) => None,
            WaveSpeedRepr3D::Grid(grid) => Some(grid.grid()),
        }
    }

    pub fn grid_dims(&self) -> Option<[usize; 3]> {
        match &self.repr {
            WaveSpeedRepr3D::Cpu(_) => None,
            WaveSpeedRepr3D::Grid(grid) => Some(grid.dims()),
        }
    }

    pub fn grid_bbox(&self) -> Option<[f32; 6]> {
        match &self.repr {
            WaveSpeedRepr3D::Cpu(_) => None,
            WaveSpeedRepr3D::Grid(grid) => Some(grid.bbox()),
        }
    }
}

impl<B: Backend> std::fmt::Debug for WaveSpeedFn3D<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WaveSpeedFn3D")
            .field("has_grid", &self.has_grid())
            .finish()
    }
}

impl<B: Backend> Module<B> for WaveSpeedFn3D<B> {
    type Record = ();

    fn collect_devices(&self, mut devices: Devices<B>) -> Devices<B> {
        if let WaveSpeedRepr3D::Grid(grid) = &self.repr {
            devices.push(grid.grid().device());
        }
        devices
    }

    fn to_device(self, device: &B::Device) -> Self {
        match self.repr {
            WaveSpeedRepr3D::Cpu(func) => Self {
                repr: WaveSpeedRepr3D::Cpu(func),
            },
            WaveSpeedRepr3D::Grid(g) => Self {
                repr: WaveSpeedRepr3D::Grid(WaveSpeedGrid3D {
                    grid: g.grid.to_device(device),
                    ..g
                }),
            },
        }
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        let _ = visitor;
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        let _ = mapper;
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {
        {}
    }
}

impl<B: Backend> burn::module::ModuleDisplayDefault for WaveSpeedFn3D<B> {
    fn content(
        &self,
        content: burn::module::Content,
    ) -> std::option::Option<burn::module::Content> {
        Some(content)
    }
}

impl<B: Backend> burn::module::ModuleDisplay for WaveSpeedFn3D<B> {}

impl<B: AutodiffBackend> AutodiffModule<B> for WaveSpeedFn3D<B> {
    type InnerModule = WaveSpeedFn3D<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        match &self.repr {
            WaveSpeedRepr3D::Cpu(func) => WaveSpeedFn3D {
                repr: WaveSpeedRepr3D::Cpu(func.clone()),
            },
            WaveSpeedRepr3D::Grid(g) => WaveSpeedFn3D {
                repr: WaveSpeedRepr3D::Grid(WaveSpeedGrid3D {
                    grid: g.grid.clone().inner(),
                    data_cpu: g.data_cpu.clone(),
                    dims: g.dims,
                    bbox: g.bbox,
                }),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_wavespeed_creation_closure() -> KwaversResult<()> {
        let wave_speed = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|_x, _y, _z| 1500.0));

        assert!(!wave_speed.has_grid());
        assert_eq!(wave_speed.get(0.0, 0.0, 0.0), 1500.0);
        Ok(())
    }

    #[test]
    fn test_wavespeed_creation_grid() -> KwaversResult<()> {
        let device = Default::default();
        let grid = Tensor::<TestBackend, 3>::ones([10, 10, 10], &device).mul_scalar(3000.0);

        let wave_speed = WaveSpeedFn3D::<TestBackend>::from_grid_with_bbox(
            grid,
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        )?;

        assert!(wave_speed.has_grid());
        Ok(())
    }

    #[test]
    fn test_wavespeed_evaluation() -> KwaversResult<()> {
        // Constant speed
        let constant = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|_x, _y, _z| 1500.0));
        assert_eq!(constant.get(0.5, 0.5, 0.5), 1500.0);
        assert_eq!(constant.get(1.0, 1.0, 1.0), 1500.0);

        // Spatially varying (layered)
        let layered = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|_x, _y, z| {
            if z < 0.5 {
                1500.0 // Water layer
            } else {
                3000.0 // Tissue layer
            }
        }));
        assert_eq!(layered.get(0.5, 0.5, 0.3), 1500.0);
        assert_eq!(layered.get(0.5, 0.5, 0.7), 3000.0);
        Ok(())
    }

    #[test]
    fn test_wavespeed_radial_variation() -> KwaversResult<()> {
        // Radially varying speed (spherical inclusion)
        let radial = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|x, y, z| {
            let r = (x * x + y * y + z * z).sqrt();
            if r < 0.5 {
                2000.0 // Inclusion
            } else {
                1500.0 // Background
            }
        }));

        assert_eq!(radial.get(0.0, 0.0, 0.0), 2000.0); // Center
        assert_eq!(radial.get(1.0, 0.0, 0.0), 1500.0); // Outside
        Ok(())
    }

    #[test]
    fn test_wavespeed_device_migration() -> KwaversResult<()> {
        let device1 = Default::default();
        let grid1 = Tensor::<TestBackend, 3>::ones([5, 5, 5], &device1);
        let wave_speed1 = WaveSpeedFn3D::<TestBackend>::from_grid_with_bbox(
            grid1,
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        )?;

        // Migrate to same device (no-op for NdArray)
        let device2 = Default::default();
        let wave_speed2 = wave_speed1.to_device(&device2);

        assert!(wave_speed2.has_grid());
        Ok(())
    }

    #[test]
    fn test_wavespeed_clone() -> KwaversResult<()> {
        let original = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|_x, _y, _z| 1500.0));
        let cloned = original.clone();

        assert_eq!(original.get(0.5, 0.5, 0.5), cloned.get(0.5, 0.5, 0.5));
        Ok(())
    }

    #[test]
    fn test_wavespeed_debug_format() -> KwaversResult<()> {
        let wave_speed = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|_x, _y, _z| 1500.0));
        let debug_str = format!("{:?}", wave_speed);

        assert!(debug_str.contains("WaveSpeedFn3D"));
        assert!(debug_str.contains("has_grid"));
        Ok(())
    }

    #[test]
    fn test_wavespeed_grid_shape() -> KwaversResult<()> {
        let device = Default::default();
        let grid = Tensor::<TestBackend, 3>::ones([32, 64, 128], &device).mul_scalar(1500.0);
        let wave_speed = WaveSpeedFn3D::<TestBackend>::from_grid_with_bbox(
            grid,
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        )?;
        assert_eq!(wave_speed.grid_dims(), Some([32, 64, 128]));
        Ok(())
    }

    #[test]
    fn test_wavespeed_grid_trilinear_interpolation() -> KwaversResult<()> {
        let device = Default::default();
        let data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let grid = Tensor::<TestBackend, 3>::from_data(
            burn::tensor::TensorData::new(data, [2, 2, 2]),
            &device,
        );
        let wave_speed = WaveSpeedFn3D::<TestBackend>::from_grid_with_bbox(
            grid,
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        )?;

        assert!((wave_speed.get(0.0, 0.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((wave_speed.get(1.0, 1.0, 1.0) - 8.0).abs() < 1e-6);
        assert!((wave_speed.get(0.5, 0.5, 0.5) - 4.5).abs() < 1e-6);
        assert!((wave_speed.get(0.5, 0.0, 0.0) - 3.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_wavespeed_grid_invalid_bbox_rejected() -> KwaversResult<()> {
        let device = Default::default();
        let grid = Tensor::<TestBackend, 3>::ones([2, 2, 2], &device);

        let result =
            WaveSpeedFn3D::<TestBackend>::from_grid_with_bbox(grid, [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
        let err = match result {
            Ok(_) => {
                return Err(KwaversError::System(SystemError::InvalidOperation {
                    operation: "from_grid_with_bbox".to_string(),
                    reason: "Expected invalid bbox to be rejected".to_string(),
                }));
            }
            Err(e) => e,
        };

        assert!(matches!(
            err,
            KwaversError::System(SystemError::InvalidConfiguration { parameter, .. })
                if parameter == "wave_speed_grid_bbox"
        ));
        Ok(())
    }

    #[test]
    fn test_wavespeed_grid_invalid_values_rejected() -> KwaversResult<()> {
        let device = Default::default();
        let data: Vec<f32> = vec![1500.0, 0.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0];
        let grid = Tensor::<TestBackend, 3>::from_data(
            burn::tensor::TensorData::new(data, [2, 2, 2]),
            &device,
        );

        let result =
            WaveSpeedFn3D::<TestBackend>::from_grid_with_bbox(grid, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let err = match result {
            Ok(_) => {
                return Err(KwaversError::System(SystemError::InvalidOperation {
                    operation: "from_grid_with_bbox".to_string(),
                    reason: "Expected non-positive wave speed values to be rejected".to_string(),
                }));
            }
            Err(e) => e,
        };

        assert!(matches!(
            err,
            KwaversError::System(SystemError::InvalidConfiguration { parameter, .. })
                if parameter == "wave_speed_grid"
        ));
        Ok(())
    }

    #[test]
    fn test_wavespeed_grid_nan_values_rejected() -> KwaversResult<()> {
        let device = Default::default();
        let data: Vec<f32> = vec![
            1500.0,
            1500.0,
            1500.0,
            1500.0,
            1500.0,
            1500.0,
            1500.0,
            f32::NAN,
        ];
        let grid = Tensor::<TestBackend, 3>::from_data(
            burn::tensor::TensorData::new(data, [2, 2, 2]),
            &device,
        );

        let result =
            WaveSpeedFn3D::<TestBackend>::from_grid_with_bbox(grid, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let err = match result {
            Ok(_) => {
                return Err(KwaversError::System(SystemError::InvalidOperation {
                    operation: "from_grid_with_bbox".to_string(),
                    reason: "Expected NaN wave speed values to be rejected".to_string(),
                }));
            }
            Err(e) => e,
        };

        assert!(matches!(
            err,
            KwaversError::System(SystemError::InvalidConfiguration { parameter, .. })
                if parameter == "wave_speed_grid"
        ));
        Ok(())
    }

    #[test]
    fn test_wavespeed_complex_heterogeneity() -> KwaversResult<()> {
        // Multi-region medium with three materials
        let complex = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|x, _y, z| {
            if z < 0.3 {
                1500.0 // Region 1: Water
            } else if z < 0.7 && x < 0.5 {
                2500.0 // Region 2: Soft tissue (left)
            } else if z < 0.7 {
                3500.0 // Region 3: Bone (right)
            } else {
                1000.0 // Region 4: Fat
            }
        }));

        assert_eq!(complex.get(0.3, 0.5, 0.2), 1500.0); // Water
        assert_eq!(complex.get(0.3, 0.5, 0.5), 2500.0); // Soft tissue
        assert_eq!(complex.get(0.7, 0.5, 0.5), 3500.0); // Bone
        assert_eq!(complex.get(0.5, 0.5, 0.8), 1000.0); // Fat
        Ok(())
    }
}
