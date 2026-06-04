//! Pre-extracted medium snapshot for the GPU PSTD adapter.
//!
//! `GpuMediumSnapshot` is private to the `gpu_pstd` module: it exists solely
//! to capture the data the GPU solver needs at construction time, avoiding the
//! need to store a `&dyn Medium` reference (lifetime / object-safety issue).

use kwavers_grid::Grid;
use kwavers_medium::Medium;

/// Per-voxel medium data extracted once at construction time.
///
/// All fields use `f32` to match the GPU buffer layout (WGSL `f32`).
/// The extraction loop performs a single pass over the voxel lattice so that
/// allocation, sound-speed, density, nonlinearity, and absorption queries are
/// interleaved in cache-friendly order.
#[derive(Debug)]
pub(super) struct GpuMediumSnapshot {
    /// Per-voxel sound speed [m/s], C-order `ix*ny*nz + iy*nz + iz`.
    pub(super) c0_flat: Vec<f32>,
    /// Per-voxel density [kg/m³].
    pub(super) rho0_flat: Vec<f32>,
    /// Per-voxel B/(2A) nonlinearity coefficient; 0 for linear media.
    pub(super) bon_a_flat: Vec<f32>,
    /// Per-voxel absorption coefficient [dB/cm]; 0 for lossless media.
    pub(super) absorption_flat: Vec<f32>,
    /// Maximum sound speed over the domain [m/s] (used as `c_ref`).
    pub(super) c_ref: f64,
    /// `true` if any voxel has a nonlinearity coefficient > 0.
    pub(super) has_nonlinear: bool,
    /// `true` if any voxel has absorption > 0.
    pub(super) has_absorption: bool,
}

impl GpuMediumSnapshot {
    /// Extract all GPU-needed medium data in a single traversal.
    ///
    /// Queries `sound_speed`, `density`, `nonlinearity`, and `absorption` for
    /// every voxel in row-major (ix, iy, iz) order to match the GPU flat-index
    /// convention `flat = ix*ny*nz + iy*nz + iz`.
    pub(super) fn from_medium<M: Medium>(grid: &Grid, medium: &M) -> Self {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let total = nx * ny * nz;

        let mut c0_flat = Vec::with_capacity(total);
        let mut rho0_flat = Vec::with_capacity(total);
        let mut bon_a_flat = Vec::with_capacity(total);
        let mut absorption_flat = Vec::with_capacity(total);

        let mut has_nonlinear = false;
        let mut has_absorption = false;

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let c = medium.sound_speed(ix, iy, iz);
                    let rho = medium.density(ix, iy, iz);
                    let nl = medium.nonlinearity(ix, iy, iz);
                    let alpha = medium.absorption(ix, iy, iz);

                    c0_flat.push(c as f32);
                    rho0_flat.push(rho as f32);
                    // B/(2A) stored as half the nonlinearity parameter
                    // (matches the GPU shader convention `precomp_bon_a`).
                    bon_a_flat.push((nl / 2.0) as f32);
                    absorption_flat.push(alpha as f32);

                    if nl > 0.0 {
                        has_nonlinear = true;
                    }
                    if alpha > 0.0 {
                        has_absorption = true;
                    }
                }
            }
        }

        let c_ref = medium.max_sound_speed();

        Self {
            c0_flat,
            rho0_flat,
            bon_a_flat,
            absorption_flat,
            c_ref,
            has_nonlinear,
            has_absorption,
        }
    }
}
