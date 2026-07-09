//! Wave speed function wrapper
//!
//! This module provides a wrapper for wave speed functions c(x, y, z), supporting
//! both CPU-based closures and device-resident grids for efficient heterogeneous
//! media modeling.
//!
//! ## Wave Speed Representations
//!
//! - **CPU Closure**: Function `c(x, y, z) → f32` evaluated on-demand
//! - **Device Grid**: Pre-computed tensor of wave speeds for fast lookup

use std::sync::Arc;

use kwavers_core::error::{KwaversError, KwaversResult, SystemError};

mod speed_fn;
#[cfg(test)]
mod tests;

#[derive(Clone)]
pub struct WaveSpeedGrid3D<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    pub(super) grid: coeus_tensor::Tensor<f32, B>,
    pub(super) data_cpu: Arc<Vec<f32>>,
    pub(super) dims: [usize; 3],
    pub(super) bbox: [f32; 6],
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> WaveSpeedGrid3D<B>
where
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
{
    /// Try new.
    /// # Errors
    /// - Returns [`KwaversError::System`] if the precondition for a System-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    pub fn try_new(grid: coeus_tensor::Tensor<f32, B>, bbox: [f32; 6]) -> KwaversResult<Self> {
        let shape = grid.shape();
        let dims = match shape {
            [nx, ny, nz] => [*nx, *ny, *nz],
            _ => {
                return Err(KwaversError::System(SystemError::InvalidConfiguration {
                    parameter: "wave_speed_grid".to_string(),
                    reason: format!("Expected wave speed grid with 3 dimensions, got {shape:?}"),
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

        let slice = grid.as_slice();

        let expected_len = nx
            .checked_mul(ny)
            .and_then(|v| v.checked_mul(nz))
            .ok_or_else(|| {
                KwaversError::System(SystemError::InvalidConfiguration {
                    parameter: "wave_speed_grid".to_string(),
                    reason: format!("Grid size overflows usize: {dims:?}"),
                })
            })?;
        if (slice.shape()[0] * slice.shape()[1] * slice.shape()[2]) != expected_len {
            return Err(KwaversError::System(SystemError::InvalidConfiguration {
                parameter: "wave_speed_grid".to_string(),
                reason: format!(
                    "Wave speed grid data length mismatch: expected {expected_len}, got {}",
                    (slice.shape()[0] * slice.shape()[1] * slice.shape()[2])
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

        let data_cpu = Arc::new(slice.iter().cloned().collect::<Vec<_>>());

        Ok(Self {
            grid,
            data_cpu,
            dims,
            bbox,
        })
    }

    pub fn grid(&self) -> &coeus_tensor::Tensor<f32, B> {
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
#[derive(Clone)]
pub struct WaveSpeedFn3D<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    pub(super) repr: WaveSpeedRepr3D<B>,
}

#[derive(Clone)]
pub(super) enum WaveSpeedRepr3D<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    Cpu(Arc<dyn Fn(f32, f32, f32) -> f32 + Send + Sync>),
    Grid(WaveSpeedGrid3D<B>),
}
