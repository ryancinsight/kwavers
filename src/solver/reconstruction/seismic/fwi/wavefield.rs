//! Wavefield modeling for FWI
//!
//! Implements forward and adjoint wavefield propagation using finite differences.
//!
//! ## Literature Reference
//! - Virieux, J., & Operto, S. (2009). "An overview of full-waveform inversion
//!   in exploration geophysics." Geophysics, 74(6), WCC1-WCC26.

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::{Array2, Array3, Zip};

/// Wavefield modeling for seismic FWI
pub struct WavefieldModeler {
    /// Spatial grid
    grid: Grid,
    /// Time step
    dt: f64,
    /// Source wavelet
    source_wavelet: Vec<f64>,
    /// Current wavefield
    wavefield: Array3<f64>,
    /// Previous wavefield (for time stepping)
    wavefield_prev: Array3<f64>,
}

impl WavefieldModeler {
    /// Create new wavefield modeler
    pub fn new(grid: Grid, dt: f64, source_wavelet: Vec<f64>) -> Self {
        let wavefield = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let wavefield_prev = Array3::zeros((grid.nx, grid.ny, grid.nz));

        Self {
            grid,
            dt,
            source_wavelet,
            wavefield,
            wavefield_prev,
        }
    }

    /// Forward modeling with given velocity model
    pub fn forward_model(&mut self, velocity_model: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let nt = self.source_wavelet.len();
        let mut seismogram = Array3::zeros((self.grid.nx, self.grid.ny, nt));

        // Reset wavefields
        self.wavefield.fill(0.0);
        self.wavefield_prev.fill(0.0);

        // Time stepping loop
        for it in 0..nt {
            // Apply source
            if it < self.source_wavelet.len() {
                // Add source at center of domain (simplified)
                let cx = self.grid.nx / 2;
                let cy = self.grid.ny / 2;
                let cz = self.grid.nz / 2;
                self.wavefield[[cx, cy, cz]] += self.source_wavelet[it];
            }

            // Update wavefield using finite differences
            let next = self.apply_stencil(&self.wavefield, &self.wavefield_prev, velocity_model);

            // Record at surface (z=0)
            for i in 0..self.grid.nx {
                for j in 0..self.grid.ny {
                    seismogram[[i, j, it]] = next[[i, j, 0]];
                }
            }

            // Update time levels
            self.wavefield_prev.assign(&self.wavefield);
            self.wavefield.assign(&next);
        }

        Ok(seismogram)
    }

    /// Adjoint modeling for gradient computation
    pub fn adjoint_model(&mut self, adjoint_source: &Array2<f64>) -> KwaversResult<Array3<f64>> {
        let nt = adjoint_source.shape()[1];
        let mut adjoint_wavefield = Array3::zeros((self.grid.nx, self.grid.ny, self.grid.nz));

        // Reset wavefields
        self.wavefield.fill(0.0);
        self.wavefield_prev.fill(0.0);

        // Backward time stepping
        for it in (0..nt).rev() {
            // Inject adjoint source at receivers (surface)
            for i in 0..self.grid.nx {
                for j in 0..self.grid.ny {
                    if i < adjoint_source.shape()[0] {
                        self.wavefield[[i, j, 0]] += adjoint_source[[i, it]];
                    }
                }
            }

            // Apply adjoint stencil (same as forward for acoustic case)
            let velocity_model = Array3::ones((self.grid.nx, self.grid.ny, self.grid.nz)) * 1500.0; // Placeholder
            let next = self.apply_stencil(&self.wavefield, &self.wavefield_prev, &velocity_model);

            // Accumulate adjoint wavefield
            adjoint_wavefield = adjoint_wavefield + &next;

            // Update time levels
            self.wavefield_prev.assign(&self.wavefield);
            self.wavefield.assign(&next);
        }

        Ok(adjoint_wavefield)
    }

    /// Apply finite difference stencil for wave equation
    /// ∂²u/∂t² = v² ∇²u
    fn apply_stencil(
        &self,
        current: &Array3<f64>,
        previous: &Array3<f64>,
        velocity: &Array3<f64>,
    ) -> Array3<f64> {
        let mut next = Array3::zeros(current.dim());
        let dt2 = self.dt * self.dt;
        let dx2 = self.grid.dx * self.grid.dx;
        let dy2 = self.grid.dy * self.grid.dy;
        let dz2 = self.grid.dz * self.grid.dz;

        // Apply 2nd-order central difference stencil
        Zip::indexed(&mut next)
            .and(current)
            .and(previous)
            .and(velocity)
            .for_each(|(i, j, k), next_val, &curr, &prev, &vel| {
                if i > 0
                    && i < self.grid.nx - 1
                    && j > 0
                    && j < self.grid.ny - 1
                    && k > 0
                    && k < self.grid.nz - 1
                {
                    // Laplacian using central differences
                    let laplacian = (current[[i + 1, j, k]] - 2.0 * curr + current[[i - 1, j, k]])
                        / dx2
                        + (current[[i, j + 1, k]] - 2.0 * curr + current[[i, j - 1, k]]) / dy2
                        + (current[[i, j, k + 1]] - 2.0 * curr + current[[i, j, k - 1]]) / dz2;

                    // Time stepping: u^{n+1} = 2u^n - u^{n-1} + (v*dt)^2 * ∇²u^n
                    *next_val = 2.0 * curr - prev + vel * vel * dt2 * laplacian;
                } else {
                    // Apply absorbing boundary (simplified)
                    *next_val = curr * 0.95; // Damping at boundaries
                }
            });

        next
    }

    /// Get current wavefield state
    pub fn get_wavefield(&self) -> &Array3<f64> {
        &self.wavefield
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavefield_propagation() {
        let grid = Grid::new(32, 32, 32, 10.0, 10.0, 10.0);
        let dt = 0.001;
        let source = vec![1.0; 100]; // Simple spike

        let mut modeler = WavefieldModeler::new(grid, dt, source);
        let velocity = Array3::ones((32, 32, 32)) * 1500.0;

        let result = modeler.forward_model(&velocity);
        assert!(result.is_ok());

        // Check that energy propagates
        let seismogram = result.unwrap();
        let total_energy: f64 = seismogram.iter().map(|x| x * x).sum();
        assert!(total_energy > 0.0);
    }
}
