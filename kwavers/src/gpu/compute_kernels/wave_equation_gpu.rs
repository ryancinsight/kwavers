//! `WaveEquationGpu`: GPU-accelerated wave equation step.

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid;
use ndarray::Array3;

use super::AcousticFieldKernel;

/// Wave equation solver on GPU.
#[derive(Debug)]
pub struct WaveEquationGpu {
    kernel: AcousticFieldKernel,
}

impl WaveEquationGpu {
    /// Create new GPU wave equation solver.
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    pub async fn new() -> KwaversResult<Self> {
        Ok(Self {
            kernel: AcousticFieldKernel::new().await?,
        })
    }

    /// Solve one coupled pressure/velocity step.
    ///
    /// ## Theorem
    /// Fusing the central-difference pressure-gradient computation with the
    /// velocity update is algebraically identical to materializing the three
    /// gradient component fields and then applying the scalar magnitude update
    /// cell by cell, because each output cell depends only on the local stencil
    /// values from the same input pressure field.
    ///
    /// ## Proof sketch
    /// The update at `(i,j,k)` reads only the six neighboring pressure samples
    /// and the current velocity sample at the same coordinates. Therefore the
    /// gradient components do not alias across cells, and the fused loop
    /// performs the same arithmetic in the same order for each interior cell.
    ///
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    pub fn step(
        &self,
        pressure: &Array3<f64>,
        velocity: &Array3<f64>,
        density: &Array3<f64>,
        sound_speed: &Array3<f64>,
        grid: &grid::Grid,
        dt: f64,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>)> {
        let (nx, ny, nz) = pressure.dim();
        if velocity.dim() != (nx, ny, nz)
            || density.dim() != (nx, ny, nz)
            || sound_speed.dim() != (nx, ny, nz)
        {
            return Err(KwaversError::InvalidInput(
                "WaveEquationGpu::step requires matching field dimensions".to_string(),
            ));
        }
        if nx < 3 || ny < 3 || nz < 3 {
            return Err(KwaversError::InvalidInput(
                "WaveEquationGpu::step requires grid dimensions >= 3".to_string(),
            ));
        }

        let c_avg = sound_speed.mean().unwrap_or(1500.0);
        let rho_avg = density.mean().unwrap_or(1000.0);

        // Update pressure: p_new = p + dt * (−ρc² ∇·v)
        let new_pressure = self.kernel.compute_propagation(pressure, grid, dt, c_avg)?;

        // Update velocity using the local pressure-gradient magnitude.
        // The scalar update contract is preserved while avoiding three full
        // temporary gradient volumes.
        let mut new_velocity = velocity.clone();
        let grad_scale = dt / rho_avg;
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let dpx = (pressure[[i + 1, j, k]] - pressure[[i - 1, j, k]]) / (2.0 * grid.dx);
                    let dpy = (pressure[[i, j + 1, k]] - pressure[[i, j - 1, k]]) / (2.0 * grid.dy);
                    let dpz = (pressure[[i, j, k + 1]] - pressure[[i, j, k - 1]]) / (2.0 * grid.dz);
                    let grad_magnitude = (dpx * dpx + dpy * dpy + dpz * dpz).sqrt();
                    new_velocity[[i, j, k]] -= grad_scale * grad_magnitude;
                }
            }
        }

        Ok((new_pressure, new_velocity))
    }
}
