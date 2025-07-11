// physics/chemistry/mod.rs
use crate::grid::Grid;
use crate::medium::Medium;
use log::debug;
use ndarray::Array3;

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

    pub fn update_chemical(
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
        debug!("Updating chemical effects");

        self.radical_initiation.update_radicals(p, light, bubble_radius, grid, dt, medium, frequency);

        if self.enable_photochemical {
            if let Some(photo) = &mut self.photochemical {
                photo.update_photochemical(
                    light,
                    emission_spectrum,
                    bubble_radius,
                    temperature,
                    grid,
                    dt,
                    medium,
                );
            }
        }

        if self.enable_kinetics {
            if let Some(kinetics) = &mut self.kinetics {
                let total_radicals = if self.enable_photochemical && self.photochemical.is_some() {
                    self.radical_initiation.radical_concentration.clone() + self.photochemical.as_ref().unwrap().reactive_oxygen_species()
                } else {
                    self.radical_initiation.radical_concentration.clone()
                };
                kinetics.update_reactions(&total_radicals, temperature, grid, dt, medium);
            }
        }
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
        // Call the inherent method, which has the same name and signature
        self.update_chemical(p, light, emission_spectrum, bubble_radius, temperature, grid, dt, medium, frequency);
    }

    fn radical_concentration(&self) -> &Array3<f64> {
        // Call the inherent method
        self.radical_concentration()
    }
}