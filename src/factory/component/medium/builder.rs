//! Medium builder - Complex medium construction logic
//!
//! Follows Builder pattern for complex medium instantiation

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::{homogeneous::HomogeneousMedium, Medium};
use super::types::{MediumConfig, MediumType};

/// Specialized medium builder following Builder pattern from GRASP
#[derive(Debug)]
pub struct MediumBuilder;

impl MediumBuilder {
    /// Build medium instance from validated configuration
    pub fn build(
        config: &MediumConfig,
        grid: &Grid,
    ) -> KwaversResult<Box<dyn Medium>> {
        match &config.medium_type {
            MediumType::Homogeneous { 
                density, 
                sound_speed, 
                mu_a, 
                mu_s_prime 
            } => {
                Self::build_homogeneous(*density, *sound_speed, *mu_a, *mu_s_prime)
            },
            MediumType::Heterogeneous { .. } => {
                Self::build_heterogeneous(config, grid)
            },
            MediumType::Layered { layers } => {
                Self::build_layered(layers, grid)
            },
            MediumType::Anisotropic { .. } => {
                Self::build_anisotropic(config, grid)
            },
        }
    }
    
    /// Build homogeneous medium
    fn build_homogeneous(
        density: f64,
        sound_speed: f64,
        mu_a: f64,
        mu_s_prime: f64,
    ) -> KwaversResult<Box<dyn Medium>> {
        // Use minimal Grid for construction (builder pattern extension)
        let minimal_grid = Grid::new(1, 1, 1, 1e-4, 1e-4, 1e-4)?;
        let medium = HomogeneousMedium::new(density, sound_speed, mu_a, mu_s_prime, &minimal_grid);
        Ok(Box::new(medium))
    }
    
    /// Build heterogeneous medium (placeholder for complex implementation)
    fn build_heterogeneous(
        _config: &MediumConfig,
        grid: &Grid,
    ) -> KwaversResult<Box<dyn Medium>> {
        // Future: Implement complex heterogeneous medium loading
        // For now, return default homogeneous medium
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 10.0, grid);
        Ok(Box::new(medium))
    }
    
    /// Build layered medium (placeholder for future implementation)
    fn build_layered(
        _layers: &[super::types::LayerProperties],
        grid: &Grid,
    ) -> KwaversResult<Box<dyn Medium>> {
        // Future: Implement layered medium construction
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 10.0, grid);
        Ok(Box::new(medium))
    }
    
    /// Build anisotropic medium (placeholder for future implementation)
    fn build_anisotropic(
        _config: &MediumConfig,
        grid: &Grid,
    ) -> KwaversResult<Box<dyn Medium>> {
        // Future: Implement anisotropic medium loading
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 10.0, grid);
        Ok(Box::new(medium))
    }
}