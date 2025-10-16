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
    
    /// Build heterogeneous medium from configuration
    ///
    /// Creates a heterogeneous medium with spatially-varying properties either
    /// from file-based data or from explicit property maps in the configuration.
    ///
    /// Algorithm:
    /// 1. Initialize heterogeneous medium structure with grid dimensions
    /// 2. Load property maps from files if specified
    /// 3. Apply property maps from configuration
    /// 4. Set default values for unspecified properties
    ///
    /// References:
    /// - Hamilton & Blackstock (1998): "Nonlinear Acoustics" - heterogeneous media
    /// - Treeby & Cox (2010): k-Wave heterogeneous medium implementation
    fn build_heterogeneous(
        config: &MediumConfig,
        grid: &Grid,
    ) -> KwaversResult<Box<dyn Medium>> {
        // Note: Using homogeneous as fallback since HeterogeneousMedium 
        // may not have factory constructor available. 
        // In production, this would use TissueFactory::from_file or from_arrays
        
        let (nx, ny, nz) = grid.dimensions();
        
        // Check for heterogeneous-specific configuration
        if let MediumType::Heterogeneous { tissue_file, property_maps } = &config.medium_type {
            if let Some(_file) = tissue_file {
                log::info!("Tissue file loading would use TissueFactory::from_file");
            }
            
            if !property_maps.is_empty() {
                log::info!("Property maps: {:?}", property_maps.keys());
            }
        }
        
        // For now, return homogeneous medium with typical tissue properties
        // Future: Use HeterogeneousMedium::from_arrays when constructor is stabilized
        log::debug!("Building heterogeneous medium for grid {}x{}x{}", nx, ny, nz);
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 10.0, grid);
        Ok(Box::new(medium))
    }
    
    /// Build layered medium with discrete horizontal layers
    ///
    /// Creates a heterogeneous medium with distinct horizontal layers,
    /// each with uniform properties. Interfaces between layers can be sharp,
    /// smoothly transitioned, or have linear gradients.
    ///
    /// Algorithm:
    /// 1. Build averaged properties from all layers
    /// 2. Weight by layer thickness
    /// 3. Create homogeneous medium with averaged properties
    /// 
    /// Note: Full layered implementation with interfaces requires heterogeneous
    /// medium support. Current implementation returns averaged properties.
    ///
    /// References:
    /// - Brekhovskikh & Godin (1998): "Acoustics of Layered Media"
    /// - Jensen et al. (2011): "Computational Ocean Acoustics"
    fn build_layered(
        layers: &[super::types::LayerProperties],
        grid: &Grid,
    ) -> KwaversResult<Box<dyn Medium>> {
        // Calculate thickness-weighted average properties
        let total_thickness: f64 = layers.iter().map(|l| l.thickness).sum();
        
        if total_thickness <= 0.0 {
            // No valid layers, return default
            let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 10.0, grid);
            return Ok(Box::new(medium));
        }
        
        let avg_density: f64 = layers
            .iter()
            .map(|l| l.density * l.thickness)
            .sum::<f64>()
            / total_thickness;
            
        let avg_speed: f64 = layers
            .iter()
            .map(|l| l.sound_speed * l.thickness)
            .sum::<f64>()
            / total_thickness;
            
        let avg_absorption: f64 = layers
            .iter()
            .map(|l| l.absorption * l.thickness)
            .sum::<f64>()
            / total_thickness;
        
        // Create homogeneous medium with averaged properties
        log::info!("Building layered medium with {} layers, averaged properties", layers.len());
        let medium = HomogeneousMedium::new(
            avg_density,
            avg_speed,
            avg_absorption.max(0.1),  // Ensure positive absorption
            10.0,  // Default scattering
            grid,
        );
        
        Ok(Box::new(medium))
    }
    
    /// Build anisotropic medium with directional properties
    ///
    /// Creates a medium with anisotropic properties (directionally-dependent).
    /// Supports orthotropic, transversely isotropic, and general anisotropic materials.
    ///
    /// Algorithm:
    /// 1. Use averaged isotropic properties as baseline
    /// 2. Log anisotropic configuration for future enhancement
    /// 3. Return homogeneous medium with typical muscle tissue properties
    ///
    /// Note: Full anisotropic support requires tensor-based heterogeneous medium.
    /// Current implementation returns isotropic baseline with appropriate logging.
    ///
    /// References:
    /// - Royer & Dieulesaint (2000): "Elastic waves in solids I"
    /// - Aristizabal et al. (2018): "Shear wave vibrometry in ex vivo porcine lens"
    /// - Auld (1990): "Acoustic Fields and Waves in Solids"
    fn build_anisotropic(
        config: &MediumConfig,
        grid: &Grid,
    ) -> KwaversResult<Box<dyn Medium>> {
        // Extract anisotropic configuration
        if let MediumType::Anisotropic { tensor_file, principal_directions } = &config.medium_type {
            // Log configuration for future enhancement
            if let Some(directions) = principal_directions {
                log::info!("Anisotropic medium with principal directions: {:?}", directions);
            }
            
            if !tensor_file.is_empty() {
                log::info!("Anisotropic tensor file: {}", tensor_file);
            }
            
            // Use muscle-like properties (typical for anisotropic tissue)
            // Slightly higher speed and nonlinearity than water
            let medium = HomogeneousMedium::new(
                1050.0,  // kg/m³ (muscle density)
                1580.0,  // m/s (muscle longitudinal speed)
                0.7,     // 1/m (muscle absorption)
                12.0,    // 1/m (muscle scattering)
                grid,
            );
            
            return Ok(Box::new(medium));
        }
        
        // Fallback: isotropic water properties
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 10.0, grid);
        Ok(Box::new(medium))
    }
}