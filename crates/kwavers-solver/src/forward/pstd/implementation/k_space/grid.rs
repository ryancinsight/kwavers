//! PSTD k-Space Grid Implementation

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_math::fft::KSpaceCalculator;
use leto::Array1;
use leto::Array3;

/// Wavenumber grid for PSTD simulations
#[derive(Debug, Clone)]
pub struct PSTDKSGrid {
    pub kx: Array1<f64>,
    pub ky: Array1<f64>,
    pub kz: Array1<f64>,
    pub k_mag: Array3<f64>,
}

impl PSTDKSGrid {
    /// Create k-space grid from spatial grid
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn from_spatial_grid(spatial_grid: &Grid) -> KwaversResult<Self> {
        let kx = Self::compute_wavenumbers(spatial_grid.nx, spatial_grid.dx);
        let ky = Self::compute_wavenumbers(spatial_grid.ny, spatial_grid.dy);
        let kz = Self::compute_wavenumbers(spatial_grid.nz, spatial_grid.dz);

        // Precompute k magnitude grid
        let mut k_mag = Array3::zeros((spatial_grid.nx, spatial_grid.ny, spatial_grid.nz));
        for i in 0..spatial_grid.nx {
            for j in 0..spatial_grid.ny {
                for k in 0..spatial_grid.nz {
                    k_mag[[i, j, k]] = kz[k]
                        .mul_add(kz[k], ky[j].mul_add(ky[j], kx[i].powi(2)))
                        .sqrt();
                }
            }
        }

        Ok(Self { kx, ky, kz, k_mag })
    }

    /// Compute wavenumber grid for one dimension using FFT conventions
    #[must_use]
    pub fn compute_wavenumbers(n: usize, dx: f64) -> Array1<f64> {
        KSpaceCalculator::generate_k_vector(n, dx).into()
    }

    /// Get grid dimensions (nx, ny, nz)
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize, usize) {
        let shape = self.k_mag.dim();
        (shape.0, shape.1, shape.2)
    }
}
