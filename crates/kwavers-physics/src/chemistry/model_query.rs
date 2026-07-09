//! ChemicalModel query/accessor methods

use super::model::ChemicalModel;
use super::parameters::ChemicalMetrics;
use leto::Array3;
use std::collections::HashMap;

impl ChemicalModel {
    /// Get radical concentrations for all tracked species
    #[must_use]
    pub fn get_radical_concentrations(&self) -> HashMap<String, Array3<f64>> {
        let mut map = HashMap::new();

        if let Some(ref kinetics) = self.kinetics {
            map.insert("OH".to_owned(), kinetics.hydroxyl_concentration().clone());
            map.insert("H2O2".to_owned(), kinetics.hydrogen_peroxide().clone());
        }

        map.insert(
            "radical_precursors".to_owned(),
            self.radical_initiation.radical_concentration.clone(),
        );

        map
    }

    /// Get reaction rates for tracked chemical reactions
    #[must_use]
    pub fn get_reaction_rates(&self) -> HashMap<String, f64> {
        let mut rates = HashMap::new();

        if let Some(ref kinetics) = self.kinetics {
            let radical_conc = &self.radical_initiation.radical_concentration;
            let oh_conc = kinetics.hydroxyl_concentration();
            let h2o2_conc = kinetics.hydrogen_peroxide();

            let avg_radical = radical_conc.mean().unwrap_or(0.0);
            let avg_oh = oh_conc.mean().unwrap_or(0.0);
            let avg_h2o2 = h2o2_conc.mean().unwrap_or(0.0);

            rates.insert(
                "OH_production_rate".to_owned(),
                avg_radical * kwavers_core::constants::thermodynamic::SONOCHEMISTRY_BASE_RATE,
            );
            rates.insert(
                "H2O2_production_rate".to_owned(),
                avg_oh * avg_oh * kwavers_core::constants::thermodynamic::SECONDARY_REACTION_RATE,
            );
            rates.insert("OH_concentration_avg".to_owned(), avg_oh);
            rates.insert("H2O2_concentration_avg".to_owned(), avg_h2o2);
        }

        rates.insert("update_count".to_owned(), self.update_count as f64);
        rates.insert(
            "reactions_total".to_owned(),
            self.metrics.total_reactions as f64,
        );

        rates
    }

    /// Get photochemical emission spectrum
    #[must_use]
    pub fn get_emission_spectrum(&self) -> Option<&Array3<f64>> {
        self.photochemical
            .as_ref()
            .map(|p| &p.reactive_oxygen_species)
    }

    /// Get performance metrics
    #[must_use]
    pub fn get_metrics(&self) -> &ChemicalMetrics {
        &self.metrics
    }
}
