use std::sync::Arc;

use kwavers_core::error::KwaversResult;

use super::super::geometry::Geometry3D;
use super::{WaveSpeedFn3D, WaveSpeedGrid3D, WaveSpeedRepr3D};

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> WaveSpeedFn3D<B>
where
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
{
    /// New.
    pub fn new(func: Arc<dyn Fn(f32, f32, f32) -> f32 + Send + Sync>) -> Self {
        Self {
            repr: WaveSpeedRepr3D::Cpu(func),
        }
    }
    /// From grid with bbox.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    pub fn from_grid_with_bbox(
        grid: coeus_tensor::Tensor<f32, B>,
        bbox: [f32; 6],
    ) -> KwaversResult<Self> {
        Ok(Self {
            repr: WaveSpeedRepr3D::Grid(WaveSpeedGrid3D::try_new(grid, bbox)?),
        })
    }
    /// From grid with geometry.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    pub fn from_grid_with_geometry(
        grid: coeus_tensor::Tensor<f32, B>,
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

    pub fn get(&self, x: f32, y: f32, z: f32) -> f32 {
        match &self.repr {
            WaveSpeedRepr3D::Cpu(func) => func(x, y, z),
            WaveSpeedRepr3D::Grid(grid) => grid.sample(x, y, z),
        }
    }

    pub fn has_grid(&self) -> bool {
        matches!(self.repr, WaveSpeedRepr3D::Grid(_))
    }

    pub fn grid_tensor(&self) -> Option<&coeus_tensor::Tensor<f32, B>> {
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

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for WaveSpeedFn3D<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WaveSpeedFn3D")
            .field("has_grid", &matches!(self.repr, WaveSpeedRepr3D::Grid(_)))
            .finish()
    }
}
