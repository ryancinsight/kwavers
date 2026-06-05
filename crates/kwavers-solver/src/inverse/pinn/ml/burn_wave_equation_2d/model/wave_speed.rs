//! `WaveSpeedFn` — spatially varying wave speed wrapper implementing the Burn `Module` trait.

use burn::module::{Module, ModuleMapper, ModuleVisitor};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;
use kwavers_core::error::{KwaversError, KwaversResult};
use std::sync::Arc;

/// Wrapper for wave speed function to implement Debug and Module traits.
#[derive(Clone)]
pub struct WaveSpeedFn<B: Backend> {
    /// CPU function for wave speed.
    pub func: Arc<dyn Fn(f32, f32) -> f32 + Send + Sync>,
    /// Optional device-resident grid of wave speeds.
    pub grid: Option<Tensor<B, 2>>,
}

impl<B: Backend> WaveSpeedFn<B> {
    /// Create a new wave speed function from a CPU closure.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(func: Arc<dyn Fn(f32, f32) -> f32 + Send + Sync>) -> Self {
        Self { func, grid: None }
    }

    /// Create a new wave speed function from a device-resident grid.
    /// # Errors
    /// - Returns [`KwaversError::System`] if the precondition for a System-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn from_grid(grid: Tensor<B, 2>) -> KwaversResult<Self> {
        let shape = grid.shape();
        let dims = match shape.dims.as_slice() {
            [nx, ny] => [*nx, *ny],
            _ => {
                return Err(KwaversError::System(
                    kwavers_core::error::SystemError::InvalidConfiguration {
                        parameter: "wave_speed_grid".to_string(),
                        reason: format!(
                            "Expected wave speed grid with 2 dimensions, got {:?}",
                            shape.dims
                        ),
                    },
                ))
            }
        };

        let [nx, ny] = dims;
        if nx == 0 || ny == 0 {
            return Err(KwaversError::System(
                kwavers_core::error::SystemError::InvalidConfiguration {
                    parameter: "wave_speed_grid".to_string(),
                    reason: format!("Grid dimensions must be non-zero, got {dims:?}"),
                },
            ));
        }

        let data = grid.clone().to_data();
        let slice = data.as_slice::<f32>().map_err(|e| {
            KwaversError::System(kwavers_core::error::SystemError::InvalidConfiguration {
                parameter: "wave_speed_grid".to_string(),
                reason: format!("Expected f32 tensor data for wave speed grid: {e:?}"),
            })
        })?;

        let expected_len = nx.checked_mul(ny).ok_or_else(|| {
            KwaversError::System(kwavers_core::error::SystemError::InvalidConfiguration {
                parameter: "wave_speed_grid".to_string(),
                reason: format!("Grid size overflows usize: {dims:?}"),
            })
        })?;
        if slice.len() != expected_len {
            return Err(KwaversError::System(
                kwavers_core::error::SystemError::InvalidConfiguration {
                    parameter: "wave_speed_grid".to_string(),
                    reason: format!(
                        "Wave speed grid data length mismatch: expected {expected_len}, got {}",
                        slice.len()
                    ),
                },
            ));
        }

        let data_cpu = Arc::new(slice.to_vec());
        let func = Arc::new(move |x: f32, y: f32| -> f32 {
            let x = x.clamp(0.0, 1.0);
            let y = y.clamp(0.0, 1.0);

            let fx = if nx <= 1 { 0.0 } else { x * ((nx - 1) as f32) };
            let fy = if ny <= 1 { 0.0 } else { y * ((ny - 1) as f32) };

            let fx0 = fx.floor();
            let fy0 = fy.floor();

            let x0 = (fx0 as isize).clamp(0, (nx - 1) as isize) as usize;
            let y0 = (fy0 as isize).clamp(0, (ny - 1) as isize) as usize;
            let x1 = (x0 + 1).min(nx - 1);
            let y1 = (y0 + 1).min(ny - 1);

            let wx = (fx - fx0).clamp(0.0, 1.0);
            let wy = (fy - fy0).clamp(0.0, 1.0);

            let at = |ix: usize, iy: usize| -> f32 { data_cpu[(ix * ny) + iy] };

            let c00 = at(x0, y0);
            let c10 = at(x1, y0);
            let c01 = at(x0, y1);
            let c11 = at(x1, y1);

            let c0 = c00 + wx * (c10 - c00);
            let c1 = c01 + wx * (c11 - c01);
            c0 + wy * (c1 - c0)
        });

        Ok(Self {
            func,
            grid: Some(grid),
        })
    }

    /// Get wave speed at coordinates (x, y).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn get(&self, x: f32, y: f32) -> f32 {
        (self.func)(x, y)
    }
}

impl<B: Backend> std::fmt::Debug for WaveSpeedFn<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WaveSpeedFn")
    }
}

impl<B: Backend> Module<B> for WaveSpeedFn<B> {
    type Record = ();

    fn collect_devices(&self, mut devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        if let Some(grid) = &self.grid {
            devices.push(grid.device());
        }
        devices
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            func: self.func,
            grid: self.grid.map(|g| g.to_device(device)),
        }
    }

    fn fork(self, device: &B::Device) -> Self {
        Self {
            func: self.func,
            grid: self.grid.map(|g| g.to_device(device)),
        }
    }

    fn map<M: ModuleMapper<B>>(self, _mapper: &mut M) -> Self {
        self
    }

    fn visit<V: ModuleVisitor<B>>(&self, _visitor: &mut V) {}

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: Backend> burn::module::ModuleDisplayDefault for WaveSpeedFn<B> {
    fn content(
        &self,
        content: burn::module::Content,
    ) -> std::option::Option<burn::module::Content> {
        Some(content)
    }
}
impl<B: Backend> burn::module::ModuleDisplay for WaveSpeedFn<B> {}

impl<B: AutodiffBackend> burn::module::AutodiffModule<B> for WaveSpeedFn<B> {
    type InnerModule = WaveSpeedFn<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        WaveSpeedFn {
            func: self.func.clone(),
            grid: self.grid.as_ref().map(|g| g.clone().inner()),
        }
    }
}
