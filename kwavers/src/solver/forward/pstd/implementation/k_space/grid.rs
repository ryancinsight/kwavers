//! PSTD k-Space Grid Implementation

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::Array1;

/// Wavenumber grid for PSTD simulations
#[derive(Debug, Clone)]
pub struct PSTDKSGrid {
    pub kx: Array1<f64>,
    pub ky: Array1<f64>,
    pub kz: Array1<f64>,
    pub k_mag: ndarray::Array3<f64>,
}

impl PSTDKSGrid {
    /// Create k-space grid from spatial grid
    pub fn from_spatial_grid(spatial_grid: &Grid) -> KwaversResult<Self> {
        let kx = Self::compute_wavenumbers(spatial_grid.nx, spatial_grid.dx);
        let ky = Self::compute_wavenumbers(spatial_grid.ny, spatial_grid.dy);
        let kz = Self::compute_wavenumbers(spatial_grid.nz, spatial_grid.dz);

        // Precompute k magnitude grid
        let mut k_mag = ndarray::Array3::zeros((spatial_grid.nx, spatial_grid.ny, spatial_grid.nz));
        for i in 0..spatial_grid.nx {
            for j in 0..spatial_grid.ny {
                for k in 0..spatial_grid.nz {
                    k_mag[[i, j, k]] = (kx[i].powi(2) + ky[j].powi(2) + kz[k].powi(2)).sqrt();
                }
            }
        }

        Ok(Self { kx, ky, kz, k_mag })
    }

    /// Compute wavenumber grid for one dimension using FFT conventions
    pub fn compute_wavenumbers(n: usize, dx: f64) -> Array1<f64> {
        let dk = 2.0 * std::f64::consts::PI / (n as f64 * dx);
        Array1::from_shape_fn(n, |i| {
            if i <= n / 2 {
                (i as f64) * dk
            } else {
                ((i as f64) - (n as f64)) * dk
            }
        })
    }

    /// Get grid dimensions (nx, ny, nz)
    pub fn dimensions(&self) -> (usize, usize, usize) {
        let shape = self.k_mag.dim();
        (shape.0, shape.1, shape.2)
    }
}
