//! `WaveEquationGpu`: GPU-accelerated wave equation step.

use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid;
use leto::Array3 as LetoArray3;

use super::{AcousticFieldKernel, AcousticFieldProvider, WgpuAcousticFieldProvider};

/// Wave equation solver on GPU.
#[derive(Debug)]
pub struct WaveEquationGpu<P = WgpuAcousticFieldProvider>
where
    P: AcousticFieldProvider,
{
    kernel: AcousticFieldKernel<P>,
}

impl WaveEquationGpu<WgpuAcousticFieldProvider> {
    /// Create new WGPU wave equation solver.
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    pub async fn new() -> KwaversResult<Self> {
        Self::try_new()
    }

    /// Create new WGPU wave equation solver synchronously.
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    pub fn try_new() -> KwaversResult<Self> {
        Ok(Self {
            kernel: AcousticFieldKernel::try_new()?,
        })
    }
}

impl<P> WaveEquationGpu<P>
where
    P: AcousticFieldProvider,
{
    /// Build a wave-equation solver from a concrete acoustic-field provider.
    #[must_use]
    pub const fn from_provider(provider: P) -> Self {
        Self {
            kernel: AcousticFieldKernel::from_provider(provider),
        }
    }

    /// Borrow the provider-backed acoustic kernel.
    #[must_use]
    pub const fn kernel(&self) -> &AcousticFieldKernel<P> {
        &self.kernel
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
        pressure: &LetoArray3<f32>,
        velocity: &LetoArray3<f32>,
        density: &LetoArray3<f32>,
        sound_speed: &LetoArray3<f32>,
        grid: &kwavers_grid::Grid,
        dt: f32,
    ) -> KwaversResult<(LetoArray3<f32>, LetoArray3<f32>)> {
        let [nx, ny, nz] = pressure.shape();
        if velocity.shape() != [nx, ny, nz]
            || density.shape() != [nx, ny, nz]
            || sound_speed.shape() != [nx, ny, nz]
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

        let c_avg = mean_or_default(sound_speed, SOUND_SPEED_WATER_SIM as f32);
        let rho_avg = mean_or_default(density, DENSITY_WATER_NOMINAL as f32);

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
                    let dpx = (pressure[[i + 1, j, k]] - pressure[[i - 1, j, k]])
                        / (2.0 * grid.dx as f32);
                    let dpy = (pressure[[i, j + 1, k]] - pressure[[i, j - 1, k]])
                        / (2.0 * grid.dy as f32);
                    let dpz = (pressure[[i, j, k + 1]] - pressure[[i, j, k - 1]])
                        / (2.0 * grid.dz as f32);
                    let grad_magnitude = (dpx * dpx + dpy * dpy + dpz * dpz).sqrt();
                    new_velocity[[i, j, k]] -= grad_scale * grad_magnitude;
                }
            }
        }

        Ok((new_pressure, new_velocity))
    }
}

fn mean_or_default(field: &LetoArray3<f32>, default: f32) -> f32 {
    let count = field.size();
    if count == 0 {
        return default;
    }

    field.iter().copied().sum::<f32>() / count as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wave_equation_wrapper_is_generic_over_acoustic_provider() {
        fn assert_provider<P>()
        where
            P: AcousticFieldProvider,
        {
            let _ = core::mem::size_of::<WaveEquationGpu<P>>();
        }

        assert_provider::<WgpuAcousticFieldProvider>();
    }
}
