//! Grid creator - Specialized grid instantiation logic
//!
//! Follows Creator pattern with optimized grid construction

use super::config::GridConfig;
use crate::error::KwaversResult;
use crate::grid::Grid;

/// Specialized grid creator following Creator pattern from GRASP
#[derive(Debug)]
pub struct GridCreator;

impl GridCreator {
    /// Create grid instance from validated configuration
    pub fn create(config: &GridConfig) -> KwaversResult<Grid> {
        // Create grid with domain information
        let grid = Grid::new(
            config.nx, config.ny, config.nz, config.dx, config.dy, config.dz,
        )?;

        Ok(grid)
    }

    /// Create grid with optimized memory layout
    ///
    /// **Implementation**: Currently uses standard row-major layout per ndarray default
    /// **Future Enhancement**: Could implement cache-optimized layouts (Z-curve, Hilbert curve)
    /// for better spatial locality in 3D traversals (Morton 1966, Hilbert 1891).
    /// Current layout provides good performance for standard FDTD/PSTD iterations.
    ///
    /// **Reference**: Morton (1966) "A Computer Oriented Geodetic Data Base"
    pub fn create_optimized(config: &GridConfig) -> KwaversResult<Grid> {
        // Standard creation provides production-ready memory layout
        Self::create(config)
    }
}
