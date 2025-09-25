//! Medium factory - Backward compatibility wrapper
//!
//! Maintains existing API while delegating to hierarchical implementation

// Re-export hierarchical components for backward compatibility
pub use super::component::medium::{MediumConfig, MediumFactory, MediumType};

// Legacy imports for constants
use crate::physics::constants::{
    DENSITY_TISSUE, DENSITY_WATER, SOUND_SPEED_TISSUE, SOUND_SPEED_WATER,
};

// Additional convenience methods for the factory
impl MediumFactory {
    /// Create water medium with standard properties
    pub fn create_water() -> crate::error::KwaversResult<Box<dyn crate::medium::Medium>> {
        use crate::medium::homogeneous::HomogeneousMedium;
        use crate::grid::Grid;
        let minimal_grid = Grid::new(1, 1, 1, 1e-4, 1e-4, 1e-4)?;
        let medium = HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.5, 10.0, &minimal_grid);
        Ok(Box::new(medium))
    }
    
    /// Create tissue medium with standard properties  
    pub fn create_tissue() -> crate::error::KwaversResult<Box<dyn crate::medium::Medium>> {
        use crate::medium::homogeneous::HomogeneousMedium;
        use crate::grid::Grid;
        let minimal_grid = Grid::new(1, 1, 1, 1e-4, 1e-4, 1e-4)?;
        let medium = HomogeneousMedium::new(DENSITY_TISSUE, SOUND_SPEED_TISSUE, 0.75, 15.0, &minimal_grid);
        Ok(Box::new(medium))
    }
}