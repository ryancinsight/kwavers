// physics/chemistry/mod.rs
use crate::grid::Grid;
use crate::medium::Medium;
use log::debug;
use ndarray::Array3;
use crate::error::KwaversResult;

/// Parameters for chemical update operations
/// Follows SOLID principles by grouping related parameters
#[derive(Debug)]
pub struct ChemicalUpdateParams<'a> {
    pub pressure: &'a Array3<f64>,
    pub light: &'a Array3<f64>,
    pub emission_spectrum: &'a Array3<f64>,
    pub bubble_radius: &'a Array3<f64>,
    pub temperature: &'a Array3<f64>,
    pub grid: &'a Grid,
    pub dt: f64,
    pub medium: &'a dyn Medium,
    pub frequency: f64,
}

impl<'a> ChemicalUpdateParams<'a> {
    /// Create new chemical update parameters
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pressure: &'a Array3<f64>,
        light: &'a Array3<f64>,
        emission_spectrum: &'a Array3<f64>,
        bubble_radius: &'a Array3<f64>,
        temperature: &'a Array3<f64>,
        grid: &'a Grid,
        dt: f64,
        medium: &'a dyn Medium,
        frequency: f64,
    ) -> Self {
        Self {
            pressure,
            light,
            emission_spectrum,
            bubble_radius,
            temperature,
            grid,
            dt,
            medium,
            frequency,
        }
    }
}

pub mod radical_initiation;
pub mod photochemistry;
pub mod reaction_kinetics;

use radical_initiation::RadicalInitiation;
use photochemistry::PhotochemicalEffects;
use reaction_kinetics::ReactionKinetics;

#[derive(Debug)]
pub struct ChemicalModel {
    radical_initiation: RadicalInitiation,
    kinetics: Option<ReactionKinetics>,
    photochemical: Option<PhotochemicalEffects>,
    #[allow(dead_code)]
    enable_kinetics: bool,
    enable_photochemical: bool,
}

impl ChemicalModel {
    pub fn new(grid: &Grid, enable_kinetics: bool, enable_photochemical: bool) -> Self {
        debug!(
            "Initializing ChemicalModel, kinetics: {}, photochemical: {}",
            enable_kinetics, enable_photochemical
        );
        Self {
            radical_initiation: RadicalInitiation::new(grid),
            kinetics: if enable_kinetics { Some(ReactionKinetics::new(grid)) } else { None },
            photochemical: if enable_photochemical {
                Some(PhotochemicalEffects::new(grid))
            } else {
                None
            },
            enable_kinetics,
            enable_photochemical,
        }
    }

    /// Update chemical effects using parameter struct
    /// Follows SOLID principles by reducing parameter coupling
    pub fn update_chemical(&mut self, params: &ChemicalUpdateParams) -> KwaversResult<()> {
        debug!("Updating chemical effects");

        self.radical_initiation.update_radicals(
            params.pressure, 
            params.light, 
            params.bubble_radius, 
            params.grid, 
            params.dt, 
            params.medium, 
            params.frequency
        );

        if self.enable_photochemical {
            if let Some(photo) = &mut self.photochemical {
                photo.update_photochemical(
                    params.light,
                    params.emission_spectrum,
                    params.bubble_radius,
                    params.temperature,
                    params.grid,
                    params.dt,
                    params.medium,
                );
            }
        }

        Ok(())
    }
    
    /// Legacy method for backward compatibility
    /// Deprecated: Use update_chemical with ChemicalUpdateParams instead
    #[deprecated(since = "0.1.0", note = "Use update_chemical with ChemicalUpdateParams instead")]
    #[allow(clippy::too_many_arguments)]
    pub fn update_chemical_legacy(
        &mut self,
        p: &Array3<f64>,
        light: &Array3<f64>,
        emission_spectrum: &Array3<f64>,
        bubble_radius: &Array3<f64>,
        temperature: &Array3<f64>,
        grid: &Grid,
        dt: f64,
        medium: &dyn Medium,
        frequency: f64,
    ) -> KwaversResult<()> {
        let params = ChemicalUpdateParams::new(
            p, light, emission_spectrum, bubble_radius, temperature,
            grid, dt, medium, frequency
        );
        self.update_chemical(&params)
    }

    pub fn radical_concentration(&self) -> &Array3<f64> {
        &self.radical_initiation.radical_concentration
    }

    pub fn hydroxyl_concentration(&self) -> Option<&Array3<f64>> {
        self.kinetics.as_ref().map(|k| k.hydroxyl_concentration())
    }

    pub fn hydrogen_peroxide(&self) -> Option<&Array3<f64>> {
        self.kinetics.as_ref().map(|k| k.hydrogen_peroxide())
    }

    pub fn reactive_oxygen_species(&self) -> Option<&Array3<f64>> {
        self.photochemical.as_ref().map(|p| p.reactive_oxygen_species())
    }
}

// ADDED:
use crate::physics::traits::ChemicalModelTrait;

impl ChemicalModelTrait for ChemicalModel {
    fn update_chemical(
        &mut self,
        p: &Array3<f64>,
        light: &Array3<f64>,
        emission_spectrum: &Array3<f64>,
        bubble_radius: &Array3<f64>,
        temperature: &Array3<f64>,
        grid: &Grid,
        dt: f64,
        medium: &dyn Medium,
        frequency: f64,
    ) {
        // Call the inherent method using the ChemicalUpdateParams struct
        let params = ChemicalUpdateParams {
            pressure: p,
            light,
            emission_spectrum,
            bubble_radius,
            temperature,
            grid,
            dt,
            medium,
            frequency,
        };
        let _ = self.update_chemical(&params);
    }

    fn radical_concentration(&self) -> &Array3<f64> {
        // Call the inherent method
        self.radical_concentration()
    }
}