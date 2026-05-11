use burn::module::{AutodiffModule, Devices, Module, ModuleMapper, ModuleVisitor};
use burn::tensor::{backend::AutodiffBackend, backend::Backend, Tensor};
use std::sync::Arc;

use crate::core::error::KwaversResult;

use super::super::geometry::Geometry3D;
use super::{WaveSpeedFn3D, WaveSpeedGrid3D, WaveSpeedRepr3D};

impl<B: Backend> WaveSpeedFn3D<B> {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(func: Arc<dyn Fn(f32, f32, f32) -> f32 + Send + Sync>) -> Self {
        Self {
            repr: WaveSpeedRepr3D::Cpu(func),
        }
    }
    /// From grid with bbox.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn from_grid_with_bbox(grid: Tensor<B, 3>, bbox: [f32; 6]) -> KwaversResult<Self> {
        Ok(Self {
            repr: WaveSpeedRepr3D::Grid(WaveSpeedGrid3D::try_new(grid, bbox)?),
        })
    }
    /// From grid with geometry.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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

    /// Grid dims.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn grid_dims(&self) -> Option<[usize; 3]> {
        match &self.repr {
            WaveSpeedRepr3D::Cpu(_) => None,
            WaveSpeedRepr3D::Grid(grid) => Some(grid.dims()),
        }
    }

    /// Grid bbox.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
