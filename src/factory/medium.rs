//! Medium factory convenience wrapper
//!
//! Re-exports hierarchical implementation with additional convenience methods

pub use super::component::medium::{MediumConfig, MediumFactory, MediumType};

use crate::physics::constants::{
    DENSITY_TISSUE, DENSITY_WATER, SOUND_SPEED_TISSUE, SOUND_SPEED_WATER,
};

// Convenience methods for the factory
impl MediumFactory {
    /// Create water medium with standard properties
    pub fn create_water() -> crate::error::KwaversResult<Box<dyn crate::medium::Medium>> {
        use crate::grid::Grid;
        use crate::medium::homogeneous::HomogeneousMedium;
        let minimal_grid = Grid::new(1, 1, 1, 1e-4, 1e-4, 1e-4)?;
        let medium =
            HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.5, 10.0, &minimal_grid);
        Ok(Box::new(medium))
    }

    /// Create tissue medium with standard properties  
    pub fn create_tissue() -> crate::error::KwaversResult<Box<dyn crate::medium::Medium>> {
        use crate::grid::Grid;
        use crate::medium::homogeneous::HomogeneousMedium;
        let minimal_grid = Grid::new(1, 1, 1, 1e-4, 1e-4, 1e-4)?;
        let medium = HomogeneousMedium::new(
            DENSITY_TISSUE,
            SOUND_SPEED_TISSUE,
            0.75,
            15.0,
            &minimal_grid,
        );
        Ok(Box::new(medium))
    }
}
