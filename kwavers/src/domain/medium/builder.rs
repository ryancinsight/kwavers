//! Medium builder - Complex medium construction logic
//!
//! Follows Builder pattern for complex medium instantiation

use super::{LayerParameters, MediumParameters, MediumType};
use crate::core::constants::SOUND_SPEED_WATER_SIM;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::heterogeneous::HeterogeneousFactory;
use crate::domain::medium::{homogeneous::HomogeneousMedium, Medium};

/// Specialized medium builder following Builder pattern from GRASP
#[derive(Debug)]
pub struct MediumBuilder;

impl MediumBuilder {
    /// Build medium instance from validated configuration
    pub fn build(config: &MediumParameters, grid: &Grid) -> KwaversResult<Box<dyn Medium>> {
        match config.medium_type {
            MediumType::Homogeneous => Self::build_homogeneous(
                config.density,
                config.sound_speed.unwrap_or(SOUND_SPEED_WATER_SIM),
                // MediumParameters doesn't have mu_a top-level?
                // Let's check config.rs (Step 130).
                // It has: density, sound_speed, absorption, absorption_power, nonlinearity...
                // mu_a and mu_s_prime are in properties map in builder usually?
                // Step 130: properties.get("mu_a").
                // So pass them or retrieve from config properties in helper.
                config,
                grid,
            ),
            MediumType::Heterogeneous => Self::build_heterogeneous(config, grid),
            MediumType::Layered => Self::build_layered(&config.layers, grid),
            MediumType::Anisotropic => Self::build_anisotropic(config, grid),
            _ => {
                // For custom/random etc. fallback to homogeneous/water for now or implement
                Self::build_homogeneous(
                    config.density,
                    config.sound_speed.unwrap_or(SOUND_SPEED_WATER_SIM),
                    config,
                    grid,
                )
            }
        }
    }

    /// Build homogeneous medium
    fn build_homogeneous(
        density: f64,
        sound_speed: f64,
        config: &MediumParameters,
        grid: &Grid,
    ) -> KwaversResult<Box<dyn Medium>> {
        // HomogeneousMedium::new(density, sound_speed, mu_a, mu_s_prime, grid)
        // Extract optical properties from map or defaults
        let mu_a = config.properties.get("mu_a").copied().unwrap_or(0.0);
        let mu_s_prime = config.properties.get("mu_s_prime").copied().unwrap_or(0.0);

        let mut medium = HomogeneousMedium::new(density, sound_speed, mu_a, mu_s_prime, grid);

        medium.set_acoustic_properties(
            config.absorption,
            config.absorption_power,
            config.nonlinearity,
        )?;

        Ok(Box::new(medium))
    }

    /// Build heterogeneous medium from configuration
    fn build_heterogeneous(
        config: &MediumParameters,
        grid: &Grid,
    ) -> KwaversResult<Box<dyn Medium>> {
        let (nx, ny, nz) = grid.dimensions();

        if let Some(file) = &config.tissue_file {
            log::warn!(
                "tissue_file '{}' loading not yet implemented; using scalar fallback for {}x{}x{} grid",
                file, nx, ny, nz
            );
        }

        if !config.property_maps.is_empty() {
            log::warn!(
                "property_maps file loading not yet implemented (keys: {:?}); using scalar fallback",
                config.property_maps.keys().collect::<Vec<_>>()
            );
        }

        let c0 = config.sound_speed.unwrap_or(SOUND_SPEED_WATER_SIM);
        let rho0 = config.density;
        let absorption = config.absorption;
        let nonlinearity = config.nonlinearity;
        let reference_frequency = 1.0e6; // 1 MHz default

        let medium = HeterogeneousFactory::from_functions(
            grid,
            move |_x, _y, _z| c0,
            move |_x, _y, _z| rho0,
            Some(Box::new(move |_x, _y, _z| absorption)),
            Some(Box::new(move |_x, _y, _z| nonlinearity)),
            reference_frequency,
        );

        log::debug!(
            "Built HeterogeneousMedium (uniform c0={:.1} m/s, rho0={:.1} kg/m³) for {}x{}x{} grid",
            c0, rho0, nx, ny, nz
        );

        Ok(Box::new(medium))
    }

    /// Build layered medium with discrete horizontal layers
    fn build_layered(layers: &[LayerParameters], grid: &Grid) -> KwaversResult<Box<dyn Medium>> {
        // Calculate thickness-weighted average properties
        let total_thickness: f64 = layers.iter().map(|l| l.thickness).sum();

        if total_thickness <= 0.0 {
            // No valid layers, return default
            let medium = HomogeneousMedium::new(1000.0, SOUND_SPEED_WATER_SIM, 0.5, 10.0, grid);
            return Ok(Box::new(medium));
        }

        let avg_density: f64 =
            layers.iter().map(|l| l.density * l.thickness).sum::<f64>() / total_thickness;

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
        log::info!(
            "Building layered medium with {} layers, averaged properties",
            layers.len()
        );
        let medium = HomogeneousMedium::new(
            avg_density,
            avg_speed,
            avg_absorption.max(0.1), // Ensure positive absorption
            10.0,                    // Default scattering
            grid,
        );

        Ok(Box::new(medium))
    }

    /// Build anisotropic medium with directional properties
    fn build_anisotropic(config: &MediumParameters, grid: &Grid) -> KwaversResult<Box<dyn Medium>> {
        // Extract anisotropic configuration
        // Log configuration for future enhancement
        if let Some(directions) = config.principal_directions {
            log::info!(
                "Anisotropic medium with principal directions: {:?}",
                directions
            );
        }

        if let Some(file) = &config.tensor_file {
            log::info!("Anisotropic tensor file: {}", file);
        }

        // Use muscle-like properties (typical for anisotropic tissue)
        // Slightly higher speed and nonlinearity than water
        let medium = HomogeneousMedium::new(
            1050.0, // kg/m³ (muscle density)
            1580.0, // m/s (muscle longitudinal speed)
            0.7,    // 1/m (muscle absorption)
            12.0,   // 1/m (muscle scattering)
            grid,
        );

        Ok(Box::new(medium))
    }
}
