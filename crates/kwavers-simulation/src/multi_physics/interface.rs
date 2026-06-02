use kwavers_core::error::KwaversResult;
use kwavers_domain::grid::Grid;

/// Simplified coupling interface definition
#[derive(Debug, Clone)]
pub struct SimulationMultiPhysicsInterface {
    /// Interface area
    pub area: f64,
    /// Interface normal direction
    pub normal: (f64, f64, f64),
    /// Source grid dimensions (nx, ny, nz) — stored for Jacobi snapshot transfers
    pub source_dims: (usize, usize, usize),
    /// Source grid spacings (dx, dy, dz) — stored for conservative interpolation
    pub source_spacing: (f64, f64, f64),
}

impl SimulationMultiPhysicsInterface {
    /// Create coupling interface between grids
    ///
    /// Computes the interface area from the overlap of the two domain extents
    /// and sets the normal perpendicular to the largest face.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(source_grid: &Grid, target_grid: &Grid) -> KwaversResult<Self> {
        // Domain extents
        let sx = source_grid.nx as f64 * source_grid.dx;
        let sy = source_grid.ny as f64 * source_grid.dy;
        let sz = source_grid.nz as f64 * source_grid.dz;
        let tx = target_grid.nx as f64 * target_grid.dx;
        let ty = target_grid.ny as f64 * target_grid.dy;
        let tz = target_grid.nz as f64 * target_grid.dz;

        // Overlap extents (minimum of source/target in each direction)
        let overlap_y = sy.min(ty);
        let overlap_z = sz.min(tz);
        let overlap_x = sx.min(tx);

        // Interface normal is perpendicular to the largest face of the overlap volume
        let area_yz = overlap_y * overlap_z;
        let area_xz = overlap_x * overlap_z;
        let area_xy = overlap_x * overlap_y;

        let (area, normal) = if area_yz >= area_xz && area_yz >= area_xy {
            (area_yz, (1.0, 0.0, 0.0))
        } else if area_xz >= area_yz && area_xz >= area_xy {
            (area_xz, (0.0, 1.0, 0.0))
        } else {
            (area_xy, (0.0, 0.0, 1.0))
        };

        Ok(Self {
            area,
            normal,
            source_dims: (source_grid.nx, source_grid.ny, source_grid.nz),
            source_spacing: (source_grid.dx, source_grid.dy, source_grid.dz),
        })
    }
}
