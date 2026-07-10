//! `WaveSpeedFn` — spatially varying wave speed lookup.

use coeus_ops::BackendOps;
use kwavers_core::error::{KwaversError, KwaversResult};
use std::sync::Arc;

/// Wrapper for a spatially varying wave speed function c(x, y).
///
/// Not a learnable parameter: this is a fixed lookup (either a closure or a
/// pre-sampled grid), never updated by gradient descent.
#[derive(Clone)]
pub struct WaveSpeedFn<B: BackendOps<f32> + Default> {
    /// CPU function for wave speed.
    pub func: Arc<dyn Fn(f32, f32) -> f32 + Send + Sync>,
    /// Optional device-resident grid of wave speeds.
    pub grid: Option<coeus_tensor::Tensor<f32, B>>,
}

impl<B: BackendOps<f32> + Default> WaveSpeedFn<B> {
    /// Create a new wave speed function from a CPU closure.
    pub fn new(func: Arc<dyn Fn(f32, f32) -> f32 + Send + Sync>) -> Self {
        Self { func, grid: None }
    }

    /// Create a new wave speed function from a device-resident grid.
    /// # Errors
    /// - Returns [`KwaversError::System`] if the precondition for a System-class constraint is violated.
    ///
    pub fn from_grid(grid: coeus_tensor::Tensor<f32, B>) -> KwaversResult<Self>
    where
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let dims = grid.shape().to_vec();
        let [nx, ny] = match dims.as_slice() {
            [nx, ny] => [*nx, *ny],
            _ => {
                return Err(KwaversError::System(
                    kwavers_core::error::SystemError::InvalidConfiguration {
                        parameter: "wave_speed_grid".to_string(),
                        reason: format!("Expected wave speed grid with 2 dimensions, got {dims:?}"),
                    },
                ))
            }
        };

        if nx == 0 || ny == 0 {
            return Err(KwaversError::System(
                kwavers_core::error::SystemError::InvalidConfiguration {
                    parameter: "wave_speed_grid".to_string(),
                    reason: format!("Grid dimensions must be non-zero, got [{nx}, {ny}]"),
                },
            ));
        }

        let slice = grid.as_slice();

        let expected_len = nx.checked_mul(ny).ok_or_else(|| {
            KwaversError::System(kwavers_core::error::SystemError::InvalidConfiguration {
                parameter: "wave_speed_grid".to_string(),
                reason: format!("Grid size overflows usize: [{nx}, {ny}]"),
            })
        })?;
        if (slice.len()) != expected_len {
            return Err(KwaversError::System(
                kwavers_core::error::SystemError::InvalidConfiguration {
                    parameter: "wave_speed_grid".to_string(),
                    reason: format!(
                        "Wave speed grid data length mismatch: expected {expected_len}, got {}",
                        (slice.len())
                    ),
                },
            ));
        }

        let data_cpu = Arc::new(slice.iter().cloned().collect::<Vec<_>>());
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
    pub fn get(&self, x: f32, y: f32) -> f32 {
        (self.func)(x, y)
    }
}

impl<B: BackendOps<f32> + Default> std::fmt::Debug for WaveSpeedFn<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WaveSpeedFn")
    }
}
