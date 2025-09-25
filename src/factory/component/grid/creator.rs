//! Grid creator - Specialized grid instantiation logic
//!
//! Follows Creator pattern with optimized grid construction

use crate::error::KwaversResult;
use crate::grid::Grid;
use super::config::GridConfig;

/// Specialized grid creator following Creator pattern from GRASP
#[derive(Debug)]
pub struct GridCreator;

impl GridCreator {
    /// Create grid instance from validated configuration
    pub fn create(config: &GridConfig) -> KwaversResult<Grid> {
        // Create grid with domain information
        let grid = Grid::new(
            config.nx,
            config.ny, 
            config.nz,
            config.dx,
            config.dy,
            config.dz,
        )?;
        
        Ok(grid)
    }
    
    /// Create grid with optimized memory layout
    pub fn create_optimized(config: &GridConfig) -> KwaversResult<Grid> {
        // Future: Add memory layout optimizations
        // For now, delegate to standard creation
        Self::create(config)
    }
}