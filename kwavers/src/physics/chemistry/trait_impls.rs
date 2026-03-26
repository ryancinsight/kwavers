//! Trait implementations for ChemicalModel

use super::model::ChemicalModel;
use super::parameters::ChemicalUpdateParams;
use super::reactions::ReactionType;
use crate::domain::grid::Grid;
use crate::physics::traits::ChemicalModelTrait;
use ndarray::Array3;

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
        medium: &dyn crate::domain::medium::Medium,
        frequency: f64,
    ) {
        let params_result = ChemicalUpdateParams::new(
            p,
            light,
            emission_spectrum,
            bubble_radius,
            temperature,
            grid,
            dt,
            medium,
            frequency,
        );

        match params_result {
            Ok(params) => {
                if let Err(e) = self.update(&params) {
                    log::error!("Chemical update failed: {}", e);
                }
            }
            Err(e) => {
                log::error!("Failed to create chemical update params: {}", e);
            }
        }
    }

    fn radical_concentration(&self) -> &Array3<f64> {
        &self.radical_initiation.radical_concentration
    }
}

impl std::fmt::Display for ReactionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            ReactionType::Dissociation => "Dissociation",
            ReactionType::Recombination => "Recombination",
            ReactionType::Oxidation => "Oxidation",
            ReactionType::Reduction => "Reduction",
            ReactionType::Polymerization => "Polymerization",
        };
        write!(f, "{name}")
    }
}
